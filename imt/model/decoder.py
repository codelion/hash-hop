"""Local decoder module that attends to retrieved chunks."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""

    def __init__(self, config: IMTConfig) -> None:
        """Initialize decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.d_model = config.d_model

        # Self-attention
        self.norm1 = nn.RMSNorm(config.d_model)
        self.self_attn = nn.MultiHeadAttention(config.d_model, config.decoder_heads)

        # Cross-attention to retrieved chunks
        self.norm2 = nn.RMSNorm(config.d_model)
        self.cross_attn = nn.MultiHeadAttention(config.d_model, config.decoder_heads)

        # Feed-forward
        self.norm3 = nn.RMSNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.decoder_ff_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ff_dim, config.d_model),
        )

        self.dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        self_attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass through decoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            context: Context from retrieved chunks of shape (batch, context_len, d_model).
            self_attn_mask: Optional causal mask for self-attention.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention
        h = self.norm1(x)
        h = self.self_attn(h, h, h, mask=self_attn_mask)
        x = x + self.dropout(h)

        # Cross-attention to retrieved chunks
        h = self.norm2(x)
        h = self.cross_attn(h, context, context)
        x = x + self.dropout(h)

        # Feed-forward
        h = self.norm3(x)
        h = self.ff(h)
        x = x + self.dropout(h)

        return x


class LocalDecoder(nn.Module):
    """Transformer decoder that attends to retrieved chunks.

    Architecture:
    1. Self-attention on query tokens
    2. Cross-attention to retrieved chunk hidden states
    3. Feed-forward network

    For HashHop, the query is the hash key we're looking up,
    and we need to produce the final value.
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize local decoder.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Query embedding (for the hash we're looking up)
        self.query_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.query_pos_embed = nn.Embedding(config.max_hash_length, config.d_model)

        # Decoder layers
        self.layers = [DecoderLayer(config) for _ in range(config.decoder_layers)]

        # Output projection
        self.norm = nn.RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def __call__(
        self,
        query_tokens: mx.array,
        retrieved_chunks: mx.array,
        retrieval_scores: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Decode the answer given query and retrieved chunks.

        Args:
            query_tokens: Hash tokens to look up of shape (batch, query_len).
            retrieved_chunks: Retrieved chunk hidden states of shape
                (batch, top_k, chunk_size, d_model).
            retrieval_scores: Optional retrieval scores of shape (batch, top_k)
                for weighted attention.

        Returns:
            logits: Output logits of shape (batch, query_len, vocab_size).
            query_repr: Query representation of shape (batch, d_model) for retrieval.
        """
        batch_size, query_len = query_tokens.shape
        _, top_k, chunk_size, d_model = retrieved_chunks.shape

        # Embed query
        positions = mx.arange(query_len)
        x = self.query_embed(query_tokens) + self.query_pos_embed(positions)

        # Flatten retrieved chunks for cross-attention
        # (batch, top_k * chunk_size, d_model)
        context = retrieved_chunks.reshape(batch_size, top_k * chunk_size, d_model)

        # Create causal mask for self-attention (optional for this task)
        # For HashHop we don't strictly need causal masking since we're not generating
        # autoregressively, but it can help regularization

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, context)

        x = self.norm(x)

        # Output logits
        logits = self.output_proj(x)

        # Query representation (mean pool) for retrieval
        query_repr = mx.mean(x, axis=1)

        return logits, query_repr

    def get_query_representation(self, query_tokens: mx.array) -> mx.array:
        """Get query representation for retrieval (without full decoding).

        Args:
            query_tokens: Hash tokens of shape (batch, query_len).

        Returns:
            Query representation of shape (batch, d_model).
        """
        positions = mx.arange(query_tokens.shape[1])
        x = self.query_embed(query_tokens) + self.query_pos_embed(positions)
        return mx.mean(x, axis=1)
