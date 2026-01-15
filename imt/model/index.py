"""Differentiable learned index search for approximate nearest neighbor retrieval."""

from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class LearnedIndexSearch(nn.Module):
    """Differentiable approximate nearest neighbor search using learned clusters.

    Key insight: Instead of hard kNN, we use soft attention over cluster centroids,
    then soft attention within selected clusters. This maintains differentiability
    for end-to-end training.

    Architecture:
    1. Cluster assignment: soft assignment to learned centroids
    2. Intra-cluster retrieval: attention-based retrieval within top-k clusters
    3. Aggregation: weighted combination of retrieved items
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize learned index search.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.num_clusters = config.num_clusters
        self.index_dim = config.index_dim
        self.d_model = config.d_model
        self.top_k = config.retrieval_top_k
        self.temperature = config.temperature

        # Learned cluster centroids
        self.centroids = mx.random.normal((config.num_clusters, config.index_dim)) * 0.1

        # Query encoder: multi-layer projection to match index key space
        # Raw embeddings need more processing to match fully-encoded index keys
        self.query_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.index_dim),
        )

        # Learnable scale for query keys (helps match index key magnitudes)
        # Initialize larger to match index key std (~0.5-0.6)
        self.query_scale = mx.array(5.0)

    def build_index(
        self,
        all_keys: mx.array,
        all_values: mx.array,
        chunk_hidden: mx.array,
    ) -> Dict[str, Any]:
        """Build the index from all chunk keys/values.

        Args:
            all_keys: Keys of shape (num_chunks, keys_per_chunk, index_dim).
            all_values: Values of shape (num_chunks, keys_per_chunk, index_dim).
            chunk_hidden: Hidden states of shape (num_chunks, chunk_size, d_model).

        Returns:
            Dictionary containing index data structures.
        """
        num_chunks = all_keys.shape[0]
        keys_per_chunk = all_keys.shape[1]

        # Flatten keys for cluster assignment
        flat_keys = all_keys.reshape(-1, self.index_dim)  # (num_chunks * keys_per_chunk, index_dim)

        # Compute soft cluster assignments for all keys
        cluster_scores = self._compute_cluster_scores(flat_keys)
        cluster_assignments = mx.softmax(cluster_scores / self.temperature, axis=-1)

        return {
            "keys": all_keys,
            "values": all_values,
            "chunk_hidden": chunk_hidden,
            "flat_keys": flat_keys,
            "cluster_assignments": cluster_assignments,
            "num_chunks": num_chunks,
            "keys_per_chunk": keys_per_chunk,
        }

    def _compute_cluster_scores(self, keys: mx.array) -> mx.array:
        """Compute similarity scores between keys and cluster centroids.

        Args:
            keys: Key vectors of shape (num_keys, index_dim).

        Returns:
            Similarity scores of shape (num_keys, num_clusters).
        """
        # Normalized dot product (cosine similarity)
        keys_norm = keys / (mx.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        centroids_norm = self.centroids / (
            mx.linalg.norm(self.centroids, axis=-1, keepdims=True) + 1e-8
        )
        return mx.matmul(keys_norm, centroids_norm.T)

    def search(
        self,
        query: mx.array,
        index: Dict[str, Any],
        query_tokens: Optional[mx.array] = None,
        chunk_tokens: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Differentiable search: retrieve relevant chunks for a query.

        Args:
            query: Query vector of shape (batch, d_model).
            index: Index built by build_index().

        Returns:
            retrieved_hidden: Retrieved chunk hidden states of shape
                (batch, top_k, chunk_size, d_model).
            retrieval_scores: Relevance scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
            all_chunk_scores: Scores for ALL chunks of shape (batch, num_chunks).
                Used for retrieval supervision during training.
        """
        batch_size = query.shape[0]
        chunk_size = index["chunk_hidden"].shape[1]
        d_model = index["chunk_hidden"].shape[2]

        # Project query to index space using deeper encoder
        query_key = self.query_encoder(query)  # (batch, index_dim)

        # Step 1: Soft cluster selection (unused but kept for potential future use)
        cluster_scores = self._compute_cluster_scores(query_key)  # (batch, num_clusters)

        # Get cluster weights (soft selection)
        cluster_weights = mx.softmax(cluster_scores / self.temperature, axis=-1)

        # Step 2: Within-cluster retrieval
        flat_keys = index["flat_keys"]  # (total_keys, index_dim)
        cluster_assignments = index["cluster_assignments"]  # (total_keys, num_clusters)

        # Use COSINE SIMILARITY to prevent score explosion
        # Normalize query and keys
        query_key_norm = query_key / (mx.linalg.norm(query_key, axis=-1, keepdims=True) + 1e-8)
        flat_keys_norm = flat_keys / (mx.linalg.norm(flat_keys, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity scores in [-1, 1]
        key_scores = mx.matmul(query_key_norm, flat_keys_norm.T)  # (batch, total_keys)

        # Step 3: Aggregate to chunk level
        num_chunks = index["num_chunks"]
        keys_per_chunk = index["keys_per_chunk"]
        chunk_scores = key_scores.reshape(batch_size, num_chunks, keys_per_chunk)

        # Max-pool over keys within each chunk
        chunk_relevance = mx.max(chunk_scores, axis=-1)  # (batch, num_chunks)

        # If query/chunk tokens provided, add token-matching boost
        # This helps bootstrap retrieval before embeddings are well-trained
        if query_tokens is not None and chunk_tokens is not None:
            token_match_score = self._compute_token_match_score(
                query_tokens, chunk_tokens, num_chunks
            )
            # Combine learned scores with token matching
            # Token matching provides strong signal, learned scores allow generalization
            chunk_relevance = chunk_relevance + token_match_score * 0.5

        # Scale to reasonable logit range
        chunk_relevance_scaled = chunk_relevance * self.query_scale

        # Soft top-k selection using softmax
        retrieval_weights = mx.softmax(chunk_relevance_scaled / self.temperature, axis=-1)

        # Get top-k chunk indices (for actual retrieval)
        sorted_indices = mx.argsort(-chunk_relevance, axis=-1)
        top_k_indices = sorted_indices[:, : self.top_k]  # (batch, top_k)

        # Retrieve chunk hidden states
        chunk_hidden = index["chunk_hidden"]  # (num_chunks, chunk_size, d_model)

        # Gather top-k chunks for each batch item
        retrieved_hidden = self._gather_chunks(chunk_hidden, top_k_indices)

        # Get retrieval scores for the selected chunks
        retrieval_scores = mx.take_along_axis(retrieval_weights, top_k_indices, axis=1)

        # Return all_chunk_scores (scaled) for retrieval supervision
        return retrieved_hidden, retrieval_scores, top_k_indices, chunk_relevance_scaled

    def _compute_token_match_score(
        self,
        query_tokens: mx.array,
        chunk_tokens: mx.array,
        num_chunks: int,
    ) -> mx.array:
        """Compute consecutive token match score between queries and chunks.

        For each query, find the maximum number of consecutive query tokens
        that match starting at some position in each chunk. This is more
        precise than just checking individual token presence.

        Args:
            query_tokens: Query tokens of shape (batch, query_len).
            chunk_tokens: All chunk tokens of shape (num_chunks, chunk_size).
            num_chunks: Number of chunks.

        Returns:
            Match scores of shape (batch, num_chunks).
        """
        batch_size, query_len = query_tokens.shape
        _, chunk_size = chunk_tokens.shape

        # For each starting position j in chunk, check how many consecutive
        # query tokens match starting at j.
        # This finds exact substring matches.

        # We need to check: for each (batch, chunk, position j),
        # does query[0:k] == chunk[j:j+k] for maximum k?

        # Compute matches for each query position against all chunk positions
        # query_tokens: (batch, query_len)
        # chunk_tokens: (num_chunks, chunk_size)

        # For position i in query, check if it matches position j+i in chunk
        # This requires comparing query[i] with chunk[j+i] for all valid j

        # Build a match score for each starting position j
        # match_score[b, c, j] = number of consecutive matches starting at j

        # Get actual query length (non-padding)
        query_mask = (query_tokens != 0).astype(mx.float32)  # (batch, query_len)
        query_lengths = mx.sum(query_mask, axis=1, keepdims=True)  # (batch, 1)

        # Check first query token matches as anchor for substring search
        first_tok = query_tokens[:, 0:1]  # (batch, 1)
        first_matches = (first_tok[:, :, None] == chunk_tokens[None, :, :])  # (batch, num_chunks, chunk_size)

        # Build scores using Python lists (convert to MLX at end)
        result_list = []
        for b in range(batch_size):
            actual_len = int(query_lengths[b, 0].item())
            if actual_len == 0:
                result_list.append([0.0] * num_chunks)
                continue

            row_scores = []
            for c in range(num_chunks):
                # Find positions where first token matches using boolean indexing
                match_mask = first_matches[b, c]  # (chunk_size,)
                mx.eval(match_mask)

                # Convert to Python list to find True positions
                match_list = match_mask.tolist()
                first_match_positions = [j for j, m in enumerate(match_list) if m]

                if len(first_match_positions) == 0:
                    row_scores.append(0.0)
                    continue

                max_match = 0.0
                for pos in first_match_positions[:10]:  # Limit iterations
                    # Check how many consecutive tokens match starting at pos
                    match_count = 0
                    for i in range(actual_len):
                        if pos + i >= chunk_size:
                            break
                        if int(query_tokens[b, i]) == int(chunk_tokens[c, pos + i]):
                            match_count += 1
                        else:
                            break
                    max_match = max(max_match, match_count / actual_len)
                row_scores.append(max_match)

            result_list.append(row_scores)

        return mx.array(result_list)

    def _gather_chunks(
        self,
        chunk_hidden: mx.array,
        indices: mx.array,
    ) -> mx.array:
        """Gather chunks by indices.

        Args:
            chunk_hidden: All chunk hidden states of shape (num_chunks, chunk_size, d_model).
            indices: Indices to gather of shape (batch, top_k).

        Returns:
            Gathered chunks of shape (batch, top_k, chunk_size, d_model).
        """
        batch_size, top_k = indices.shape
        chunk_size = chunk_hidden.shape[1]
        d_model = chunk_hidden.shape[2]

        # Flatten indices and gather
        flat_indices = indices.reshape(-1)
        gathered = mx.take(chunk_hidden, flat_indices, axis=0)

        return gathered.reshape(batch_size, top_k, chunk_size, d_model)

    def compute_retrieval_supervision_loss(
        self,
        retrieval_scores: mx.array,
        chunk_indices: mx.array,
        target_chunk_indices: mx.array,
    ) -> mx.array:
        """Compute loss to encourage retrieving correct chunks.

        Args:
            retrieval_scores: Scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
            target_chunk_indices: Ground truth chunk indices of shape (batch,).

        Returns:
            Scalar loss value.
        """
        # Check if target chunk is in retrieved chunks
        target_expanded = target_chunk_indices[:, None]  # (batch, 1)
        is_target = (chunk_indices == target_expanded).astype(mx.float32)  # (batch, top_k)

        # Maximize score of target chunk, minimize others
        target_scores = (retrieval_scores * is_target).sum(axis=-1)
        non_target_scores = (retrieval_scores * (1 - is_target)).sum(axis=-1)

        # Margin loss
        margin = 0.1
        loss = mx.maximum(mx.array(0.0), non_target_scores - target_scores + margin)

        return mx.mean(loss)
