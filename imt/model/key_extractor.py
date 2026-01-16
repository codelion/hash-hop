"""Index key extractor module for learning to extract searchable keys from chunks."""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class IndexKeyExtractor(nn.Module):
    """Learns to extract searchable keys from chunk hidden states.

    Uses a combination of:
    1. Learned cross-attention with query vectors for position-specific keys
    2. Global pooling for content-based keys (helps with query-key alignment)

    The global component uses the same character embeddings as queries,
    making it easier to learn retrieval from scratch.
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize key extractor.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.keys_per_chunk = config.keys_per_chunk
        self.d_model = config.d_model
        self.index_dim = config.index_dim
        self.num_heads = config.num_index_heads

        # Learned query vectors for key extraction
        # These learn to attend to "important" positions (hash starts)
        self.key_queries = mx.random.normal((config.keys_per_chunk, config.d_model)) * 0.02

        # Projections for cross-attention
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Project to index dimension
        self.key_proj = nn.Linear(config.d_model, config.index_dim)
        self.value_proj = nn.Linear(config.d_model, config.index_dim)

        # Global content key - single key that captures overall chunk content
        # This helps with initial alignment since it pools over all positions
        self.global_key_proj = nn.Linear(config.d_model, config.index_dim)

        self.norm = nn.RMSNorm(config.d_model)

    def __call__(
        self,
        chunk_hidden: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Extract index keys and values from chunk hidden states.

        Args:
            chunk_hidden: Hidden states of shape (batch, chunk_size, d_model).

        Returns:
            keys: Searchable keys of shape (batch, keys_per_chunk, index_dim).
                First key is global content key, rest are position-specific.
            values: Associated values of shape (batch, keys_per_chunk, index_dim).
            attention_weights: Attention weights of shape (batch, keys_per_chunk, chunk_size)
                for interpretability.
        """
        batch_size = chunk_hidden.shape[0]

        # Global content key - mean pool over all positions
        # This creates a key that captures the overall content of the chunk
        global_content = mx.mean(chunk_hidden, axis=1)  # (batch, d_model)
        global_key = self.global_key_proj(global_content)  # (batch, index_dim)
        global_key = global_key[:, None, :]  # (batch, 1, index_dim)

        # Position-specific keys via cross-attention (for remaining keys)
        if self.keys_per_chunk > 1:
            # Expand key queries for batch
            queries = mx.broadcast_to(
                self.key_queries[None, :self.keys_per_chunk-1, :],
                (batch_size, self.keys_per_chunk - 1, self.d_model),
            )

            # Cross-attend to chunk hidden states
            extracted, attn_weights = self._cross_attend_with_weights(queries, chunk_hidden)
            extracted = self.norm(extracted)

            # Project to index space
            position_keys = self.key_proj(extracted)
            position_values = self.value_proj(extracted)

            # Combine global and position keys
            keys = mx.concatenate([global_key, position_keys], axis=1)
            values = mx.concatenate([global_key, position_values], axis=1)

            # Add uniform attention for global key
            uniform_attn = mx.ones((batch_size, 1, chunk_hidden.shape[1])) / chunk_hidden.shape[1]
            attn_weights = mx.concatenate([uniform_attn, attn_weights], axis=1)
        else:
            keys = global_key
            values = global_key
            attn_weights = mx.ones((batch_size, 1, chunk_hidden.shape[1])) / chunk_hidden.shape[1]

        return keys, values, attn_weights

    def _cross_attend_with_weights(
        self,
        queries: mx.array,
        context: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Cross-attention that also returns attention weights.

        Args:
            queries: Query vectors of shape (batch, num_queries, d_model).
            context: Context vectors of shape (batch, context_len, d_model).

        Returns:
            output: Attended values of shape (batch, num_queries, d_model).
            weights: Attention weights of shape (batch, num_queries, context_len).
        """
        batch_size, num_queries, _ = queries.shape
        _, context_len, _ = context.shape
        head_dim = self.d_model // self.num_heads

        # Project queries, keys, values
        q = self.q_proj(queries)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, num_heads, seq, head_dim)
        q = q.reshape(batch_size, num_queries, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, context_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, context_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = head_dim**-0.5
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        output = mx.matmul(weights, v)

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, num_queries, self.d_model)
        output = self.out_proj(output)

        # Average attention weights across heads for interpretability
        avg_weights = mx.mean(weights, axis=1)

        return output, avg_weights
