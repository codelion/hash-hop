"""Complete Indexed Memory Transformer model for HashHop."""

from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig
from imt.model.chunk_encoder import ChunkEncoder
from imt.model.decoder import LocalDecoder
from imt.model.index import LearnedIndexSearch
from imt.model.key_extractor import IndexKeyExtractor


class IndexedMemoryTransformer(nn.Module):
    """Complete Indexed Memory Transformer for HashHop.

    Architecture:
    1. ChunkEncoder: Process each 512-token chunk independently
    2. IndexKeyExtractor: Extract searchable keys from chunks
    3. LearnedIndexSearch: Build differentiable index, perform retrieval
    4. LocalDecoder: Attend to retrieved chunks, produce answer

    Training flow:
    - Encode all chunks (can be batched)
    - Extract keys from all chunks
    - Build index
    - For each query: retrieve -> decode -> loss

    Inference flow:
    - Pre-encode and index all chunks (one-time)
    - For each query: retrieve -> decode
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize IMT model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        self.encoder = ChunkEncoder(config)
        self.key_extractor = IndexKeyExtractor(config)
        self.index_search = LearnedIndexSearch(config)
        self.decoder = LocalDecoder(config)

    def encode_chunks(
        self,
        chunk_tokens: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Encode all chunks and extract index keys.

        For memory efficiency, processes chunks in batches.

        Args:
            chunk_tokens: All chunk tokens of shape (num_chunks, chunk_size).

        Returns:
            chunk_hidden: Hidden states of shape (num_chunks, chunk_size, d_model).
            index_keys: Index keys of shape (num_chunks, keys_per_chunk, index_dim).
            index_values: Index values of shape (num_chunks, keys_per_chunk, index_dim).
        """
        num_chunks = chunk_tokens.shape[0]
        batch_size = self.config.chunk_batch_size

        all_hidden = []
        all_keys = []
        all_values = []

        # Process chunks in batches
        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunk_tokens[i : i + batch_size]

            # Encode
            hidden = self.encoder(batch_chunks)

            # Extract keys
            keys, values, _ = self.key_extractor(hidden)

            all_hidden.append(hidden)
            all_keys.append(keys)
            all_values.append(values)

        # Concatenate
        chunk_hidden = mx.concatenate(all_hidden, axis=0)
        index_keys = mx.concatenate(all_keys, axis=0)
        index_values = mx.concatenate(all_values, axis=0)

        return chunk_hidden, index_keys, index_values

    def build_index(
        self,
        chunk_hidden: mx.array,
        index_keys: mx.array,
        index_values: mx.array,
    ) -> Dict[str, Any]:
        """Build the searchable index from encoded chunks.

        Args:
            chunk_hidden: Hidden states of shape (num_chunks, chunk_size, d_model).
            index_keys: Index keys of shape (num_chunks, keys_per_chunk, index_dim).
            index_values: Index values of shape (num_chunks, keys_per_chunk, index_dim).

        Returns:
            Index dictionary for use with search.
        """
        return self.index_search.build_index(index_keys, index_values, chunk_hidden)

    def forward_query(
        self,
        query_tokens: mx.array,
        index: Dict[str, Any],
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Process queries against the index.

        Args:
            query_tokens: Query hash tokens of shape (batch, query_len).
            index: Index built by build_index().

        Returns:
            logits: Output logits of shape (batch, query_len, vocab_size).
            retrieval_scores: Retrieval scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
            all_chunk_scores: Scores for ALL chunks (batch, num_chunks) for supervision.
        """
        # Initial query embedding for retrieval
        query_repr = self.decoder.get_query_representation(query_tokens)

        # Retrieve relevant chunks
        retrieved_hidden, retrieval_scores, chunk_indices, all_chunk_scores = (
            self.index_search.search(query_repr, index)
        )

        # Decode answer
        logits, _ = self.decoder(query_tokens, retrieved_hidden, retrieval_scores)

        return logits, retrieval_scores, chunk_indices, all_chunk_scores

    def __call__(
        self,
        chunk_tokens: mx.array,
        query_tokens: mx.array,
        precomputed_index: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Full forward pass.

        If precomputed_index is provided, skip encoding phase.

        Args:
            chunk_tokens: All chunk tokens of shape (num_chunks, chunk_size).
            query_tokens: Query hash tokens of shape (batch, query_len).
            precomputed_index: Optional pre-built index to skip encoding.

        Returns:
            logits: Output logits of shape (batch, query_len, vocab_size).
            retrieval_scores: Retrieval scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
            index_keys: Index keys of shape (num_chunks, keys_per_chunk, index_dim).
            all_chunk_scores: Scores for ALL chunks (batch, num_chunks) for supervision.
        """
        if precomputed_index is None:
            # Encode and index
            chunk_hidden, index_keys, index_values = self.encode_chunks(chunk_tokens)
            index = self.build_index(chunk_hidden, index_keys, index_values)
        else:
            index = precomputed_index
            index_keys = index["keys"]

        # Process queries
        logits, retrieval_scores, chunk_indices, all_chunk_scores = self.forward_query(
            query_tokens, index
        )

        return logits, retrieval_scores, chunk_indices, index_keys, all_chunk_scores

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Total parameter count.
        """
        def count_dict_params(d: dict) -> int:
            total = 0
            for v in d.values():
                if isinstance(v, mx.array):
                    total += v.size
                elif isinstance(v, dict):
                    total += count_dict_params(v)
            return total

        return count_dict_params(self.parameters())
