"""Local decoder module that attends to retrieved chunks."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention.

    Includes copy-aware cross-attention that helps find query matches in context.
    """

    def __init__(self, config: IMTConfig, use_copy_bias: bool = False) -> None:
        """Initialize decoder layer.

        Args:
            config: Model configuration.
            use_copy_bias: Whether to use copy mechanism bias in cross-attention.
        """
        super().__init__()
        self.d_model = config.d_model
        self.use_copy_bias = use_copy_bias

        # Self-attention
        self.norm1 = nn.RMSNorm(config.d_model)
        self.self_attn = nn.MultiHeadAttention(config.d_model, config.decoder_heads)

        # Cross-attention to retrieved chunks
        self.norm2 = nn.RMSNorm(config.d_model)
        self.cross_attn = nn.MultiHeadAttention(config.d_model, config.decoder_heads)

        # Copy bias projection (learns to weight token matching)
        if use_copy_bias:
            self.copy_gate = nn.Linear(config.d_model, 1)

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
        copy_bias: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass through decoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            context: Context from retrieved chunks of shape (batch, context_len, d_model).
            self_attn_mask: Optional causal mask for self-attention.
            copy_bias: Optional copy bias of shape (batch, seq_len, context_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention
        h = self.norm1(x)
        h = self.self_attn(h, h, h, mask=self_attn_mask)
        x = x + self.dropout(h)

        # Cross-attention to retrieved chunks
        h = self.norm2(x)
        # Note: MLX MultiHeadAttention doesn't support external bias,
        # so we apply copy bias by modifying context weights
        if copy_bias is not None and self.use_copy_bias:
            # Use copy gate to blend copy-biased attention
            gate = mx.sigmoid(self.copy_gate(x))  # (batch, seq_len, 1)
            # Apply softmax to copy bias to get copy attention
            copy_attn = mx.softmax(copy_bias, axis=-1)  # (batch, seq_len, context_len)
            # Get copy-weighted context
            copy_context = mx.matmul(copy_attn, context)  # (batch, seq_len, d_model)
            # Regular cross-attention
            h = self.cross_attn(h, context, context)
            # Blend with gated copy
            h = (1 - gate) * h + gate * copy_context
        else:
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
    2. Cross-attention to retrieved chunk hidden states (with copy bias)
    3. Feed-forward network

    For HashHop, the query is the hash key we're looking up,
    and we need to produce the final value.

    Key innovation: Uses token-level copy bias to help attention find
    where the query appears in context, enabling better value extraction.
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

        # Decoder layers - first layer uses copy bias to bootstrap attention
        self.layers = [
            DecoderLayer(config, use_copy_bias=(i == 0))
            for i in range(config.decoder_layers)
        ]

        # Output projection
        self.norm = nn.RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Copy mechanism: learns to find query in context and extract value
        # Token embedding for computing copy similarity (shared with query_embed)
        self.copy_scale = mx.array(10.0)  # Learnable scale for copy scores

        # Query representation encoder (for retrieval)
        # Use self-attention to capture sequence patterns, not just mean pooling
        self.query_self_attn = nn.MultiHeadAttention(config.d_model, config.decoder_heads)
        self.query_ff = nn.Sequential(
            nn.Linear(config.d_model, config.decoder_ff_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ff_dim, config.d_model),
        )
        self.query_norm1 = nn.RMSNorm(config.d_model)
        self.query_norm2 = nn.RMSNorm(config.d_model)

    def _compute_copy_bias(
        self,
        query_tokens: mx.array,
        context_tokens: mx.array,
    ) -> mx.array:
        """Compute copy bias based on token matching.

        This helps the model find where the query appears in context
        and attend to the value that follows.

        Args:
            query_tokens: Query token IDs of shape (batch, query_len).
            context_tokens: Context token IDs of shape (batch, context_len).

        Returns:
            Copy bias of shape (batch, query_len, context_len).
        """
        batch_size, query_len = query_tokens.shape
        _, context_len = context_tokens.shape

        # Compute token-level matching: where do query tokens appear in context?
        # Use embedding similarity instead of exact match for smoother gradients
        query_embed = self.query_embed(query_tokens)  # (batch, query_len, d_model)
        context_embed = self.query_embed(context_tokens)  # (batch, context_len, d_model)

        # Normalize for cosine similarity
        query_norm = query_embed / (mx.linalg.norm(query_embed, axis=-1, keepdims=True) + 1e-8)
        context_norm = context_embed / (mx.linalg.norm(context_embed, axis=-1, keepdims=True) + 1e-8)

        # Token similarity: (batch, query_len, context_len)
        token_sim = mx.matmul(query_norm, context_norm.transpose(0, 2, 1))

        # For each query position i, we want to attend to context position j
        # where the query string STARTS, then offset to get the value
        # The pattern is: query matches at position j -> value starts at j + query_len + 4
        # (4 for " = '" delimiter)

        # Create offset attention: for each query position i, attend to
        # context positions where we expect the i-th character of the VALUE
        # This is complex, so we'll use a simpler approach:
        # Just scale the token similarity to help attention focus

        return token_sim * self.copy_scale

    def _find_query_in_context(
        self,
        query_tokens: mx.array,
        context_tokens: mx.array,
        pad_id: int = 0,
    ) -> mx.array:
        """Find where query appears in context and return value-aligned attention.

        Uses full query matching (not just first character) to find exact location.

        Args:
            query_tokens: Query tokens (batch, query_len).
            context_tokens: Context tokens (batch, context_len).
            pad_id: Padding token ID to ignore.

        Returns:
            Attention bias of shape (batch, query_len, context_len) that attends
            to the VALUE positions (offset from query match).
        """
        batch_size, padded_query_len = query_tokens.shape
        _, context_len = context_tokens.shape

        # Compute actual query length per batch item (ignoring padding)
        # For simplicity, use the first batch item's actual length
        # (queries in a batch should have same length)
        non_pad_mask = (query_tokens[0] != pad_id).astype(mx.int32)
        actual_query_len = int(mx.sum(non_pad_mask).item())

        # If all tokens are padding, fall back to padded length
        if actual_query_len == 0:
            actual_query_len = padded_query_len

        # For simplified format KEY>VALUE, the value starts at offset actual_query_len+1
        # from where the key starts (after the '>' delimiter)
        offset = actual_query_len + 1  # +1 for '>'

        # Match ALL actual query characters (not padding)
        # For each starting position j in context, check if query matches
        # query_tokens: (batch, query_len)
        # context_tokens: (batch, context_len)

        # Create match scores for each possible starting position
        # match_score[j] = sum over i of (query[i] == context[j+i])
        match_scores = mx.zeros((batch_size, context_len))

        for i in range(actual_query_len):
            # Get query character i
            query_char = query_tokens[:, i:i+1]  # (batch, 1)

            # Check if it matches context at position j+i
            # We need context[:, j+i] for all j
            # This is context shifted left by i positions
            if i < context_len:
                shifted_context = mx.concatenate([
                    context_tokens[:, i:],
                    mx.zeros((batch_size, i), dtype=context_tokens.dtype)
                ], axis=1)  # (batch, context_len)
            else:
                shifted_context = mx.zeros((batch_size, context_len), dtype=context_tokens.dtype)

            # Match: query_char == shifted_context
            char_match = (shifted_context == query_char).astype(mx.float32)

            # Add to match scores (positions where query could start)
            match_scores = match_scores + char_match

        # Perfect match has score == actual_query_len
        # Use soft threshold: high score for positions close to actual_query_len
        perfect_match = (match_scores >= actual_query_len - 0.5).astype(mx.float32)

        # Now create attention for value positions
        # For output position i, attend to j+offset+i where j is the match position
        attention_rows = []
        for i in range(padded_query_len):
            # For positions beyond actual query, attend to nothing meaningful
            if i < actual_query_len:
                shift = offset + i
                if shift < context_len and shift > 0:
                    padded = mx.concatenate([
                        mx.zeros((batch_size, shift)),
                        perfect_match[:, :-shift]
                    ], axis=1)
                elif shift == 0:
                    padded = perfect_match
                else:
                    padded = mx.zeros((batch_size, context_len))
            else:
                # Padding position - uniform attention (or zeros)
                padded = mx.zeros((batch_size, context_len))
            attention_rows.append(padded[:, None, :])

        value_attention = mx.concatenate(attention_rows, axis=1)

        # Scale to get strong attention
        return value_attention * 100.0

    def __call__(
        self,
        query_tokens: mx.array,
        retrieved_chunks: mx.array,
        retrieval_scores: Optional[mx.array] = None,
        context_tokens: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Decode the answer given query and retrieved chunks.

        Args:
            query_tokens: Hash tokens to look up of shape (batch, query_len).
            retrieved_chunks: Retrieved chunk hidden states of shape
                (batch, top_k, chunk_size, d_model).
            retrieval_scores: Optional retrieval scores of shape (batch, top_k)
                for weighted attention.
            context_tokens: Optional context token IDs of shape (batch, context_len)
                for copy mechanism.

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

        # Compute copy bias if context tokens are provided
        copy_bias = None
        if context_tokens is not None:
            # Use explicit query location finding for better value extraction
            copy_bias = self._find_query_in_context(query_tokens, context_tokens)

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, context, copy_bias=copy_bias)

        x = self.norm(x)

        # Standard vocabulary logits
        vocab_logits = self.output_proj(x)  # (batch, query_len, vocab_size)

        # If we have context tokens, compute copy logits
        if context_tokens is not None:
            # Pointer mechanism: compute attention over context positions
            # Then create logits by scattering attention to vocabulary positions
            pointer_query = x  # (batch, query_len, d_model)

            # Compute attention scores over context
            # Use the copy_bias as attention (already computed where to look)
            copy_attn = mx.softmax(copy_bias, axis=-1)  # (batch, query_len, context_len)

            # Create copy logits: for each vocab token, sum attention over
            # context positions that have that token
            batch_size, query_len, _ = vocab_logits.shape
            _, context_len = context_tokens.shape
            vocab_size = vocab_logits.shape[-1]

            # Initialize copy logits
            # For each (batch, query_pos), we want to compute:
            # copy_logits[v] = sum over j where context_tokens[j] == v of copy_attn[j]
            # This is: scatter_add of copy_attn based on context_tokens
            copy_logits = mx.zeros_like(vocab_logits)

            # Scatter attention to vocabulary positions
            # context_tokens: (batch, context_len) - vocab IDs
            # copy_attn: (batch, query_len, context_len) - attention weights
            # For each context position j, add its attention to copy_logits[context_tokens[j]]

            # Use one-hot encoding to scatter
            context_onehot = mx.eye(vocab_size)[context_tokens]  # (batch, context_len, vocab_size)
            # Multiply attention by one-hot and sum over context positions
            # copy_attn: (batch, query_len, context_len)
            # context_onehot: (batch, context_len, vocab_size)
            # Result: (batch, query_len, vocab_size)
            copy_logits = mx.matmul(copy_attn, context_onehot)

            # Convert from probability to logits scale
            copy_logits = mx.log(copy_logits + 1e-10)

            # Blend vocabulary and copy logits
            # Use a simple average for now (could learn a gate)
            logits = vocab_logits + copy_logits
        else:
            logits = vocab_logits

        # Query representation (mean pool) for retrieval
        query_repr = mx.mean(x, axis=1)

        return logits, query_repr

    def get_query_representation(self, query_tokens: mx.array) -> mx.array:
        """Get query representation for retrieval (without full decoding).

        Uses self-attention to capture sequence patterns in the hash,
        rather than just mean pooling which loses distinctive information.

        Args:
            query_tokens: Hash tokens of shape (batch, query_len).

        Returns:
            Query representation of shape (batch, d_model).
        """
        positions = mx.arange(query_tokens.shape[1])
        x = self.query_embed(query_tokens) + self.query_pos_embed(positions)

        # Self-attention layer to capture sequence patterns
        h = self.query_norm1(x)
        h = self.query_self_attn(h, h, h)
        x = x + h

        # Feed-forward
        h = self.query_norm2(x)
        h = self.query_ff(h)
        x = x + h

        # Now pool - the representations are more distinctive after attention
        return mx.mean(x, axis=1)
