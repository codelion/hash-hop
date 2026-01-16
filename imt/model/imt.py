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
    4. LocalDecoder: Autoregressive decoder with copy mechanism

    Key design: Encoder and decoder share the same token embedding.
    This helps with query-key alignment since the same characters
    produce the same initial representations.

    Training flow:
    - Encode all chunks (can be batched)
    - Extract keys from all chunks
    - Build index
    - For each query: retrieve -> decode with teacher forcing -> loss

    Inference flow:
    - Pre-encode and index all chunks (one-time)
    - For each query: retrieve -> autoregressive decode
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize IMT model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Shared token embedding for encoder and decoder
        # This helps with query-key alignment
        self.shared_token_embed = nn.Embedding(config.vocab_size, config.d_model)

        self.encoder = ChunkEncoder(config)
        self.key_extractor = IndexKeyExtractor(config)
        self.index_search = LearnedIndexSearch(config)
        self.decoder = LocalDecoder(config)

        # Share embeddings: make encoder and decoder use the same token embedding
        # This is critical for learning query-key alignment from scratch
        self.encoder.token_embed = self.shared_token_embed
        self.decoder.query_embed = self.shared_token_embed

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
        target_tokens: mx.array,
        index: Dict[str, Any],
        chunk_tokens: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Process queries against the index with teacher forcing.

        Args:
            query_tokens: Query hash tokens of shape (batch, query_len).
            target_tokens: Target answer tokens of shape (batch, target_len).
            index: Index built by build_index().
            chunk_tokens: Chunk token IDs (num_chunks, chunk_size) for
                embedding-based retrieval boosting and copy mechanism.

        Returns:
            logits: Output logits of shape (batch, target_len, vocab_size).
            copy_attention: Copy attention weights (batch, target_len, context_len).
            copy_gate: Copy gate values (batch, target_len, 1).
            retrieval_scores: Retrieval scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
            all_chunk_scores: Scores for ALL chunks (batch, num_chunks) for supervision.
        """
        batch_size = query_tokens.shape[0]

        # Initial query embedding for retrieval
        query_repr = self.decoder.get_query_representation(query_tokens)

        # Compute raw embedding means for embedding-based retrieval boost
        query_embed = self.shared_token_embed(query_tokens)  # (batch, query_len, d_model)
        query_embed_mean = mx.mean(query_embed, axis=1)  # (batch, d_model)

        chunk_embed = self.shared_token_embed(chunk_tokens)  # (num_chunks, chunk_size, d_model)
        chunk_embed_means = mx.mean(chunk_embed, axis=1)  # (num_chunks, d_model)

        # Retrieve relevant chunks with embedding boost
        retrieved_hidden, retrieval_scores, chunk_indices, all_chunk_scores = (
            self.index_search.search(
                query_repr, index,
                query_embed_mean=query_embed_mean,
                chunk_embed_means=chunk_embed_means
            )
        )

        # Get chunk token IDs for retrieved chunks (for copy mechanism)
        # chunk_indices: (batch, top_k)
        top_k = chunk_indices.shape[1]
        chunk_size = chunk_tokens.shape[1]

        # Gather retrieved chunk token IDs
        retrieved_chunk_tokens = self._gather_chunk_tokens(chunk_tokens, chunk_indices)
        # Shape: (batch, top_k, chunk_size)

        # Decode answer using teacher forcing with copy mechanism
        logits, copy_attention, copy_gate = self.decoder(
            query_tokens, retrieved_hidden, target_tokens, retrieved_chunk_tokens
        )

        return logits, copy_attention, copy_gate, retrieval_scores, chunk_indices, all_chunk_scores

    def _gather_chunk_tokens(
        self,
        chunk_tokens: mx.array,
        indices: mx.array,
    ) -> mx.array:
        """Gather chunk tokens by indices.

        Args:
            chunk_tokens: All chunk tokens of shape (num_chunks, chunk_size).
            indices: Indices to gather of shape (batch, top_k).

        Returns:
            Gathered chunk tokens of shape (batch, top_k, chunk_size).
        """
        batch_size, top_k = indices.shape
        chunk_size = chunk_tokens.shape[1]

        flat_indices = indices.reshape(-1)
        gathered = mx.take(chunk_tokens, flat_indices, axis=0)

        return gathered.reshape(batch_size, top_k, chunk_size)

    def __call__(
        self,
        chunk_tokens: mx.array,
        query_tokens: mx.array,
        target_tokens: mx.array,
        precomputed_index: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Full forward pass with teacher forcing.

        If precomputed_index is provided, skip encoding phase.

        Args:
            chunk_tokens: All chunk tokens of shape (num_chunks, chunk_size).
            query_tokens: Query hash tokens of shape (batch, query_len).
            target_tokens: Target answer tokens of shape (batch, target_len).
            precomputed_index: Optional pre-built index to skip encoding.

        Returns:
            logits: Output logits of shape (batch, target_len, vocab_size).
            copy_attention: Copy attention weights (batch, target_len, context_len).
            copy_gate: Copy gate values (batch, target_len, 1).
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

        # Process queries with teacher forcing and copy mechanism
        logits, copy_attention, copy_gate, retrieval_scores, chunk_indices, all_chunk_scores = (
            self.forward_query(query_tokens, target_tokens, index, chunk_tokens)
        )

        return logits, copy_attention, copy_gate, retrieval_scores, chunk_indices, index_keys, all_chunk_scores

    def generate(
        self,
        chunk_tokens: mx.array,
        query_tokens: mx.array,
        precomputed_index: Optional[Dict[str, Any]] = None,
        max_length: int = 20,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Generate answer autoregressively (for inference).

        Args:
            chunk_tokens: All chunk tokens of shape (num_chunks, chunk_size).
            query_tokens: Query hash tokens of shape (batch, query_len).
            precomputed_index: Optional pre-built index to skip encoding.
            max_length: Maximum generation length.

        Returns:
            generated: Generated token IDs of shape (batch, gen_len).
            retrieval_scores: Retrieval scores of shape (batch, top_k).
            chunk_indices: Retrieved chunk indices of shape (batch, top_k).
        """
        if precomputed_index is None:
            # Encode and index
            chunk_hidden, index_keys, index_values = self.encode_chunks(chunk_tokens)
            index = self.build_index(chunk_hidden, index_keys, index_values)
        else:
            index = precomputed_index

        # Get query representation for retrieval
        query_repr = self.decoder.get_query_representation(query_tokens)

        # Compute embedding means for retrieval boost
        query_embed = self.shared_token_embed(query_tokens)
        query_embed_mean = mx.mean(query_embed, axis=1)

        chunk_embed = self.shared_token_embed(chunk_tokens)
        chunk_embed_means = mx.mean(chunk_embed, axis=1)

        # Retrieve relevant chunks
        retrieved_hidden, retrieval_scores, chunk_indices, _ = (
            self.index_search.search(
                query_repr, index,
                query_embed_mean=query_embed_mean,
                chunk_embed_means=chunk_embed_means
            )
        )

        # Get chunk token IDs for copy mechanism
        retrieved_chunk_tokens = self._gather_chunk_tokens(chunk_tokens, chunk_indices)

        # Generate autoregressively
        generated = self.decoder.generate(
            query_tokens, retrieved_hidden, retrieved_chunk_tokens, max_length
        )

        return generated, retrieval_scores, chunk_indices

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
