"""Chunk encoder module for processing fixed-size token chunks."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class TransformerEncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        """Initialize encoder layer.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask.

        Returns:
            Output tensor of same shape as input.
        """
        # Self-attention with pre-norm
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + self.dropout(h)

        # Feed-forward with pre-norm
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)

        return x


class ChunkEncoder(nn.Module):
    """Small transformer that encodes fixed-size chunks independently.

    Processes 512-token chunks to produce chunk representations
    and candidate index keys. Each chunk is encoded independently,
    allowing parallel processing and memory efficiency.
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize chunk encoder.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.chunk_size, config.d_model)

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(
                d_model=config.d_model,
                num_heads=config.encoder_heads,
                ff_dim=config.encoder_ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.encoder_layers)
        ]

        self.norm = nn.RMSNorm(config.d_model)

    def __call__(
        self,
        chunk_tokens: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode a batch of chunks.

        Args:
            chunk_tokens: Token indices of shape (batch, chunk_size).
            mask: Optional attention mask.

        Returns:
            Chunk hidden states of shape (batch, chunk_size, d_model).
        """
        batch_size, seq_len = chunk_tokens.shape

        # Embeddings
        positions = mx.arange(seq_len)
        x = self.token_embed(chunk_tokens) + self.pos_embed(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.norm(x)

    def encode_batched(
        self,
        all_chunks: mx.array,
        batch_size: Optional[int] = None,
    ) -> mx.array:
        """Encode all chunks in batches for memory efficiency.

        Args:
            all_chunks: All chunk tokens of shape (num_chunks, chunk_size).
            batch_size: Batch size for processing. Uses config default if None.

        Returns:
            All chunk hidden states of shape (num_chunks, chunk_size, d_model).
        """
        if batch_size is None:
            batch_size = self.config.chunk_batch_size

        num_chunks = all_chunks.shape[0]
        all_hidden = []

        for i in range(0, num_chunks, batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            hidden = self(batch_chunks)
            all_hidden.append(hidden)

        return mx.concatenate(all_hidden, axis=0)
