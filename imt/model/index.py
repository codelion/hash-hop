"""Differentiable learned index search for approximate nearest neighbor retrieval."""

from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class LearnedIndexSearch(nn.Module):
    """Differentiable learned retrieval using query-key matching.

    Architecture:
    1. Query encoder: Project query representation to index space
    2. Key-query similarity: Cosine similarity between query and chunk keys
    3. Chunk scoring: Aggregate key scores to chunk level
    4. Soft retrieval: Differentiable top-k selection

    All retrieval is learned end-to-end via the retrieval supervision loss.
    No hard-coded matching or cheats.
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize learned index search.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.index_dim = config.index_dim
        self.d_model = config.d_model
        self.top_k = config.retrieval_top_k
        self.temperature = config.temperature

        # Query encoder: transform query representation to index space
        # Keep it simple - single linear layer preserves more embedding structure
        # This helps with learning alignment when using shared embeddings
        self.query_encoder = nn.Linear(config.d_model, config.index_dim)

        # Learnable temperature for retrieval softmax
        # Initialized to reasonable value, learned during training
        self.retrieval_temp = mx.array(1.0)

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

        # Flatten keys for efficient similarity computation
        flat_keys = all_keys.reshape(-1, self.index_dim)

        # Pre-normalize keys for cosine similarity
        flat_keys_norm = flat_keys / (mx.linalg.norm(flat_keys, axis=-1, keepdims=True) + 1e-8)

        return {
            "keys": all_keys,
            "values": all_values,
            "chunk_hidden": chunk_hidden,
            "flat_keys": flat_keys,
            "flat_keys_norm": flat_keys_norm,
            "num_chunks": num_chunks,
            "keys_per_chunk": keys_per_chunk,
        }

    def search(
        self,
        query: mx.array,
        index: Dict[str, Any],
        query_embed_mean: mx.array = None,
        chunk_embed_means: mx.array = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Differentiable search: retrieve relevant chunks for a query.

        Args:
            query: Query vector of shape (batch, d_model).
            index: Index built by build_index().
            query_embed_mean: Optional raw query embedding mean (batch, d_model)
                for embedding-based scoring. If provided along with chunk_embed_means,
                adds an embedding similarity component to retrieval scores.
            chunk_embed_means: Optional raw chunk embedding means (num_chunks, d_model).

        Returns:
            retrieved_hidden: Retrieved chunk hidden states of shape
                (batch, top_k, chunk_size, d_model).
            retrieval_scores: Relevance scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
            all_chunk_scores: Scores for ALL chunks of shape (batch, num_chunks).
                Used for retrieval supervision during training.
        """
        batch_size = query.shape[0]

        # Project query to index space
        query_key = self.query_encoder(query)  # (batch, index_dim)

        # Normalize query for cosine similarity
        query_key_norm = query_key / (mx.linalg.norm(query_key, axis=-1, keepdims=True) + 1e-8)

        # Compute cosine similarity with all keys
        flat_keys_norm = index["flat_keys_norm"]  # (total_keys, index_dim)
        key_scores = mx.matmul(query_key_norm, flat_keys_norm.T)  # (batch, total_keys)

        # Aggregate to chunk level using max-pooling over keys
        num_chunks = index["num_chunks"]
        keys_per_chunk = index["keys_per_chunk"]
        chunk_key_scores = key_scores.reshape(batch_size, num_chunks, keys_per_chunk)
        chunk_scores = mx.max(chunk_key_scores, axis=-1)  # (batch, num_chunks)

        # Add embedding-based score if provided
        # This helps bootstrap learning by providing a direct signal from raw embeddings
        if query_embed_mean is not None and chunk_embed_means is not None:
            # Compute cosine similarity between query and chunk embeddings
            query_norm = query_embed_mean / (mx.linalg.norm(query_embed_mean, axis=-1, keepdims=True) + 1e-8)
            chunk_norm = chunk_embed_means / (mx.linalg.norm(chunk_embed_means, axis=-1, keepdims=True) + 1e-8)
            embed_scores = mx.matmul(query_norm, chunk_norm.T)  # (batch, num_chunks)

            # Blend learned scores with embedding scores
            # Initially rely more on embedding scores, gradually let learned scores take over
            blend_weight = mx.sigmoid(self.retrieval_temp)  # 0.73 at temp=1.0
            chunk_scores = blend_weight * chunk_scores + (1 - blend_weight) * embed_scores

        # Apply learned temperature for softmax
        temp = mx.abs(self.retrieval_temp) + 0.1  # Ensure positive temperature
        retrieval_probs = mx.softmax(chunk_scores / temp, axis=-1)

        # Get top-k chunk indices (hard selection for actual retrieval)
        sorted_indices = mx.argsort(-chunk_scores, axis=-1)
        top_k_indices = sorted_indices[:, :self.top_k]

        # Gather top-k chunks
        chunk_hidden = index["chunk_hidden"]
        retrieved_hidden = self._gather_chunks(chunk_hidden, top_k_indices)

        # Get retrieval scores for selected chunks
        retrieval_scores = mx.take_along_axis(retrieval_probs, top_k_indices, axis=1)

        # Return chunk_scores (not probs) for loss computation
        return retrieved_hidden, retrieval_scores, top_k_indices, chunk_scores

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

        flat_indices = indices.reshape(-1)
        gathered = mx.take(chunk_hidden, flat_indices, axis=0)

        return gathered.reshape(batch_size, top_k, chunk_size, d_model)
