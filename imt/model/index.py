"""Differentiable learned index search for approximate nearest neighbor retrieval."""

from typing import Any, Dict, Tuple

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
        # Scale to match index key magnitudes
        query_key = query_key * self.query_scale

        # Step 1: Soft cluster selection
        cluster_scores = self._compute_cluster_scores(query_key)  # (batch, num_clusters)

        # Get cluster weights (soft selection)
        cluster_weights = mx.softmax(cluster_scores / self.temperature, axis=-1)

        # Step 2: Within-cluster retrieval
        flat_keys = index["flat_keys"]  # (total_keys, index_dim)
        cluster_assignments = index["cluster_assignments"]  # (total_keys, num_clusters)

        # Key relevance = dot product (not normalized - allows magnitude to matter)
        # This gives stronger gradient signal than cosine similarity
        key_scores = mx.matmul(query_key, flat_keys.T)  # (batch, total_keys)

        # Step 3: Aggregate to chunk level (skip cluster weighting for simplicity)
        num_chunks = index["num_chunks"]
        keys_per_chunk = index["keys_per_chunk"]
        chunk_scores = key_scores.reshape(batch_size, num_chunks, keys_per_chunk)

        # Max-pool over keys within each chunk
        chunk_relevance = mx.max(chunk_scores, axis=-1)  # (batch, num_chunks)

        # Normalize scores to have zero mean (prevents any chunk from dominating)
        # This is critical for stable training
        chunk_relevance_centered = chunk_relevance - mx.mean(chunk_relevance, axis=-1, keepdims=True)

        # Scale for reasonable softmax behavior
        chunk_relevance_scaled = chunk_relevance_centered / self.temperature

        # Soft top-k selection using softmax with low temperature
        retrieval_weights = mx.softmax(chunk_relevance_scaled / (self.temperature * 0.1), axis=-1)

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
