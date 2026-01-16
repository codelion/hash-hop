"""Autoregressive decoder module with copy mechanism for HashHop retrieval."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from imt.config import IMTConfig


class DecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention and cross-attention.

    Uses standard transformer decoder architecture with:
    - Causal (masked) self-attention for autoregressive generation
    - Cross-attention to retrieved context chunks
    - Feed-forward network
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.d_model = config.d_model

        # Causal self-attention
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
        causal_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass through decoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            context: Context from retrieved chunks of shape (batch, context_len, d_model).
            causal_mask: Causal attention mask of shape (seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Causal self-attention
        h = self.norm1(x)
        h = self.self_attn(h, h, h, mask=causal_mask)
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
    """Autoregressive decoder with copy mechanism for HashHop.

    Architecture:
    1. Input: Query tokens (key to look up) + target tokens (for teacher forcing)
    2. Causal self-attention on decoder input
    3. Cross-attention to retrieved chunk hidden states
    4. Pointer network for copy mechanism
    5. Blend copy logits with vocabulary logits

    The copy mechanism is essential for HashHop because the model needs to
    extract exact token sequences from the retrieved context.

    Training uses teacher forcing: feed ground truth previous tokens.
    Inference uses autoregressive generation: generate one token at a time.
    """

    def __init__(self, config: IMTConfig) -> None:
        """Initialize local decoder.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # Special token IDs (must match tokenizer)
        # ASCIITokenizer uses: PAD=0, UNK=1, BOS=2, EOS=3
        self.bos_id = 2
        self.eos_id = 3
        self.pad_id = 0

        # Token embedding for decoder input (will be shared with encoder)
        self.query_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_hash_length * 2, config.d_model)

        # Decoder layers
        self.layers = [DecoderLayer(config) for _ in range(config.decoder_layers)]

        # Output projection for vocabulary logits
        self.norm = nn.RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Copy mechanism components
        # Projects decoder hidden state to query for copy attention
        self.copy_query_proj = nn.Linear(config.d_model, config.d_model)
        # Projects context hidden states to keys for copy attention
        self.copy_key_proj = nn.Linear(config.d_model, config.d_model)
        # Gate to blend copy vs vocabulary logits
        self.copy_gate_proj = nn.Linear(config.d_model, 1)

        # Query representation encoder (for retrieval)
        self.query_self_attn = nn.MultiHeadAttention(config.d_model, config.decoder_heads)
        self.query_ff = nn.Sequential(
            nn.Linear(config.d_model, config.decoder_ff_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ff_dim, config.d_model),
        )
        self.query_norm1 = nn.RMSNorm(config.d_model)
        self.query_norm2 = nn.RMSNorm(config.d_model)

    def _create_causal_mask(self, seq_len: int) -> mx.array:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length.

        Returns:
            Causal mask of shape (seq_len, seq_len) where True means attend.
        """
        # Create lower triangular mask (True = can attend, False = cannot)
        mask = mx.tril(mx.ones((seq_len, seq_len)))
        # Convert to additive mask: 0 for valid, -inf for invalid
        return mx.where(mask, 0.0, -1e9)

    def _prepare_decoder_input(
        self,
        query_tokens: mx.array,
        target_tokens: mx.array,
    ) -> mx.array:
        """Prepare decoder input by concatenating query and shifted target.

        For training with teacher forcing, we input:
        [query_tokens, BOS, target[0], target[1], ..., target[n-1]]

        And predict:
        [target[0], target[1], ..., target[n], EOS] (at positions after query)

        Args:
            query_tokens: Query hash tokens (batch, query_len).
            target_tokens: Target answer tokens (batch, target_len).

        Returns:
            Decoder input tokens (batch, query_len + target_len).
        """
        batch_size = query_tokens.shape[0]

        # Create BOS token
        bos = mx.full((batch_size, 1), self.bos_id, dtype=mx.int32)

        # Shift target: [BOS, target[:-1]]
        shifted_target = mx.concatenate([bos, target_tokens[:, :-1]], axis=1)

        # Concatenate query with shifted target
        # The model sees the query first, then generates the answer
        return mx.concatenate([query_tokens, shifted_target], axis=1)

    def _compute_copy_attention(
        self,
        hidden: mx.array,
        context: mx.array,
        context_token_ids: mx.array = None,
        query_token_ids: mx.array = None,
        output_position: int = None,
    ) -> mx.array:
        """Compute copy attention over context.

        The copy mechanism needs to attend to VALUE positions in context where
        format is KEY>VALUE. We help the model by:
        1. Computing learned attention over context positions
        2. Adding bias towards positions immediately after where KEY appears

        Args:
            hidden: Decoder hidden states (batch, seq_len, d_model).
            context: Context hidden states (batch, context_len, d_model).
            context_token_ids: Token IDs of context (batch, context_len).
            query_token_ids: Token IDs of query (batch, query_len).
            output_position: Current output position index for single-step generation.

        Returns:
            Copy attention weights (batch, seq_len, context_len).
        """
        batch_size, seq_len, _ = hidden.shape
        context_len = context.shape[1]

        # Project to query and key spaces
        query = self.copy_query_proj(hidden)  # (batch, seq_len, d_model)
        key = self.copy_key_proj(context)  # (batch, context_len, d_model)

        # Compute attention scores with scaled dot-product
        d_k = self.d_model ** 0.5
        scores = mx.matmul(query, key.transpose(0, 2, 1)) / d_k  # (batch, seq_len, context_len)

        # Add positional bias based on query token locations
        # The VALUE we want to copy comes AFTER the KEY in context
        if context_token_ids is not None and query_token_ids is not None:
            query_len = query_token_ids.shape[1]

            # For each context position, check if it matches any query token
            # query_token_ids: (batch, query_len)
            # context_token_ids: (batch, context_len)
            query_expanded = query_token_ids[:, :, None]  # (batch, query_len, 1)
            context_expanded = context_token_ids[:, None, :]  # (batch, 1, context_len)

            # Match matrix: (batch, query_len, context_len)
            matches = (query_expanded == context_expanded).astype(mx.float32)

            # Find where query sequence starts in context
            # If context[i:i+query_len] matches query, that's where KEY is
            # VALUE starts at position i + query_len + 1 (after the '>' separator)

            # Simplified: for each output position j, boost attention to positions
            # that are offset from query token matches by (query_len + 1 + j)
            # This means output position 0 attends to first char of VALUE,
            # output position 1 attends to second char, etc.

            # Find where the FULL query sequence matches in context
            # This is more reliable than just matching the first token
            # matches shape: (batch, query_len, context_len)

            # For position p in context, compute how many consecutive tokens match
            # A full match at position p means matches[:, 0, p] AND matches[:, 1, p+1] AND ...

            # We'll compute a "sequence match score" at each context position
            # sequence_match[p] = product of matches at positions p, p+1, ..., p+query_len-1
            # But products are hard with gradients, so use sum instead

            # Sum of matches across query positions, shifted appropriately
            # If context[p:p+query_len] matches query, then sequence_match[p] = query_len
            query_len_actual = min(query_len, 4)  # Only check first 4 chars (key length)

            sequence_match = mx.zeros((batch_size, context_len))
            for q_pos in range(query_len_actual):
                # Shift matches[:, q_pos, :] left by q_pos positions
                if q_pos < context_len:
                    shifted_match = mx.concatenate([
                        matches[:, q_pos, q_pos:],
                        mx.zeros((batch_size, q_pos))
                    ], axis=1)
                    sequence_match = sequence_match + shifted_match

            # Normalize to [0, 1] range
            sequence_match = sequence_match / query_len_actual

            # Positions with high sequence_match (~1) are where the query starts
            # The VALUE starts at (query_start + query_len + 1)
            # query_len here is the actual KEY length (4 chars typically)
            value_offset = query_len_actual + 1  # +1 for '>' separator

            # Build position bias list for each output position
            # If output_position is provided (single-step generation), only compute
            # bias for that specific position
            if output_position is not None:
                # During generation: only one position, use explicit output_position
                total_offset = value_offset + output_position

                if total_offset < context_len and output_position < self.config.max_hash_length:
                    zeros_before = mx.zeros((batch_size, total_offset))
                    match_truncated = sequence_match[:, :context_len - total_offset]
                    position_bias = mx.concatenate([zeros_before, match_truncated], axis=1)
                else:
                    position_bias = mx.zeros((batch_size, context_len))

                # Expand to (batch, 1, context_len) for single-position
                position_bias = position_bias[:, None, :]
            else:
                # During training: compute for all output positions
                bias_list = []
                for out_pos in range(seq_len):
                    # Total offset: where value starts + output position
                    total_offset = value_offset + out_pos

                    if total_offset < context_len and out_pos < self.config.max_hash_length:
                        # Shift sequence_match by total_offset positions
                        zeros_before = mx.zeros((batch_size, total_offset))
                        match_truncated = sequence_match[:, :context_len - total_offset]
                        shifted = mx.concatenate([zeros_before, match_truncated], axis=1)
                    else:
                        shifted = mx.zeros((batch_size, context_len))

                    bias_list.append(shifted)

                # Stack to create (batch, seq_len, context_len)
                position_bias = mx.stack(bias_list, axis=1)

            # Add position bias to scores - use strong bias
            scores = scores + 5.0 * position_bias

        # Softmax over context positions
        return mx.softmax(scores, axis=-1)

    def _scatter_copy_logits(
        self,
        copy_attention: mx.array,
        context_token_ids: mx.array,
    ) -> mx.array:
        """Convert copy attention to vocabulary logits in log space.

        For each output position, scatter the copy attention weights
        to their corresponding vocabulary indices using MAX (not sum).

        Using max instead of sum prevents common tokens from being artificially
        boosted when they appear multiple times in context.

        CRITICAL: We convert to log space so copy_logits have similar scale
        to vocab_logits. Without this, the blending doesn't work properly
        because attention weights are in [0,1] while vocab_logits are unbounded.

        Args:
            copy_attention: Copy attention (batch, seq_len, context_len).
            context_token_ids: Token IDs of context (batch, context_len).

        Returns:
            Copy logits (batch, seq_len, vocab_size) in log space.
        """
        batch_size, seq_len, context_len = copy_attention.shape

        # Build mask for each vocab token: (batch, context_len, vocab_size)
        # vocab_indices: (1, 1, vocab_size)
        vocab_indices = mx.arange(self.vocab_size)[None, None, :]
        # context_ids_expanded: (batch, context_len, 1)
        context_ids_expanded = context_token_ids[:, :, None]
        # mask: 1.0 where context position has this vocab token, 0.0 otherwise
        mask = (context_ids_expanded == vocab_indices).astype(mx.float32)

        # Expand copy_attention for broadcasting: (batch, seq_len, context_len, 1)
        attn_expanded = copy_attention[:, :, :, None]

        # Expand mask: (batch, 1, context_len, vocab_size)
        mask_expanded = mask[:, None, :, :]

        # Mask attention: set to -inf where token doesn't match
        # This ensures max only considers positions with matching token
        # attn_expanded: (batch, seq_len, context_len, 1)
        # mask_expanded: (batch, 1, context_len, vocab_size)
        # Broadcasting gives: (batch, seq_len, context_len, vocab_size)
        masked_attn = mx.where(
            mask_expanded > 0,
            attn_expanded,
            mx.array(-1e9)
        )

        # Take max over context positions: (batch, seq_len, vocab_size)
        max_attn = mx.max(masked_attn, axis=2)

        # Convert to log space with small epsilon to avoid log(0)
        # This ensures copy_logits have similar scale to vocab_logits
        # Tokens with no match in context will have log(-1e9) = very negative
        copy_logits = mx.log(mx.maximum(max_attn, 1e-10))

        return copy_logits

    def __call__(
        self,
        query_tokens: mx.array,
        retrieved_chunks: mx.array,
        target_tokens: mx.array,
        chunk_token_ids: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with teacher forcing for training.

        Args:
            query_tokens: Query hash tokens (batch, query_len).
            retrieved_chunks: Retrieved chunk hidden states
                (batch, top_k, chunk_size, d_model).
            target_tokens: Target answer tokens (batch, target_len).
            chunk_token_ids: Token IDs of retrieved chunks
                (batch, top_k, chunk_size) for copy mechanism.

        Returns:
            logits: Output logits (batch, target_len, vocab_size).
            copy_attention: Copy attention weights (batch, target_len, context_len).
            copy_gate: Copy gate values (batch, target_len, 1).
        """
        batch_size, query_len = query_tokens.shape
        _, top_k, chunk_size, d_model = retrieved_chunks.shape
        target_len = target_tokens.shape[1]

        # Prepare decoder input: [query, BOS, target[:-1]]
        decoder_input = self._prepare_decoder_input(query_tokens, target_tokens)
        total_len = decoder_input.shape[1]

        # Embed decoder input
        positions = mx.arange(total_len)
        x = self.query_embed(decoder_input) + self.pos_embed(positions)

        # Flatten retrieved chunks for cross-attention
        context = retrieved_chunks.reshape(batch_size, top_k * chunk_size, d_model)
        context_tokens = chunk_token_ids.reshape(batch_size, top_k * chunk_size)

        # Create causal mask for the target portion
        # Query tokens can see each other, but target tokens are causal
        causal_mask = self._create_causal_mask(total_len)

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, context, causal_mask)

        x = self.norm(x)

        # Extract only the target portion (after query tokens)
        target_hidden = x[:, query_len:, :]  # (batch, target_len, d_model)

        # Vocabulary logits
        vocab_logits = self.output_proj(target_hidden)  # (batch, target_len, vocab_size)

        # Copy mechanism with token-based alignment
        # Pass query_tokens to help align attention to positions after the query
        copy_attention = self._compute_copy_attention(
            target_hidden, context,
            context_token_ids=context_tokens,
            query_token_ids=query_tokens,
        )
        copy_logits = self._scatter_copy_logits(copy_attention, context_tokens)

        # Copy gate: probability of copying vs generating from vocabulary
        copy_gate = mx.sigmoid(self.copy_gate_proj(target_hidden))  # (batch, target_len, 1)

        # Blend copy and vocab logits
        # Higher copy_gate = more copying from context
        final_logits = copy_gate * copy_logits + (1 - copy_gate) * vocab_logits

        return final_logits, copy_attention, copy_gate

    def generate(
        self,
        query_tokens: mx.array,
        retrieved_chunks: mx.array,
        chunk_token_ids: mx.array,
        max_length: int = 20,
    ) -> mx.array:
        """Autoregressive generation for inference.

        Args:
            query_tokens: Query hash tokens (batch, query_len).
            retrieved_chunks: Retrieved chunk hidden states
                (batch, top_k, chunk_size, d_model).
            chunk_token_ids: Token IDs of retrieved chunks
                (batch, top_k, chunk_size).
            max_length: Maximum generation length.

        Returns:
            Generated token IDs (batch, gen_len).
        """
        batch_size, query_len = query_tokens.shape
        _, top_k, chunk_size, d_model = retrieved_chunks.shape

        # Flatten context
        context = retrieved_chunks.reshape(batch_size, top_k * chunk_size, d_model)
        context_tokens = chunk_token_ids.reshape(batch_size, top_k * chunk_size)

        # Start with BOS token
        generated = mx.full((batch_size, 1), self.bos_id, dtype=mx.int32)

        for step in range(max_length):
            # Prepare input: [query, generated_so_far]
            decoder_input = mx.concatenate([query_tokens, generated], axis=1)
            total_len = decoder_input.shape[1]

            # Embed
            positions = mx.arange(total_len)
            x = self.query_embed(decoder_input) + self.pos_embed(positions)

            # Causal mask
            causal_mask = self._create_causal_mask(total_len)

            # Process through layers
            for layer in self.layers:
                x = layer(x, context, causal_mask)

            x = self.norm(x)

            # Get logits for last position only
            last_hidden = x[:, -1:, :]  # (batch, 1, d_model)

            # Vocab logits
            vocab_logits = self.output_proj(last_hidden)

            # Copy mechanism for last position
            # Pass token IDs for position bias AND the current output position
            # step=0 means we're generating output position 0, etc.
            copy_attention = self._compute_copy_attention(
                last_hidden, context,
                context_token_ids=context_tokens,
                query_token_ids=query_tokens,
                output_position=step,  # Critical: tells position bias which char to target
            )
            copy_logits = self._scatter_copy_logits(copy_attention, context_tokens)
            copy_gate = mx.sigmoid(self.copy_gate_proj(last_hidden))

            # Blend
            final_logits = copy_gate * copy_logits + (1 - copy_gate) * vocab_logits

            # Greedy decoding: pick most likely token
            next_token = mx.argmax(final_logits[:, -1, :], axis=-1, keepdims=True)

            # Append to generated
            generated = mx.concatenate([generated, next_token], axis=1)

            # Stop if all sequences have generated EOS
            if mx.all(next_token.flatten() == self.eos_id):
                break

        # Remove BOS token
        return generated[:, 1:]

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
        x = self.query_embed(query_tokens) + self.pos_embed(positions)

        # Self-attention layer to capture sequence patterns
        h = self.query_norm1(x)
        h = self.query_self_attn(h, h, h)
        x = x + h

        # Feed-forward
        h = self.query_norm2(x)
        h = self.query_ff(h)
        x = x + h

        # Pool to get fixed-size representation
        return mx.mean(x, axis=1)
