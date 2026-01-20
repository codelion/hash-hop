"""Long-Term Memory (LTM) Model - Unified Retrieval + LLM.

Inspired by MagicLabs LTM-2: trains retrieval and generation together
so the model learns to attend over massive context (10M+ tokens).

Architecture:
1. Memory Encoder: Encodes code snippets into memory slots
2. Cross-Attention: Query attends over all memory slots
3. Decoder: Generates output based on attended memory

Key insight: Use symbolic tokenization (like HashHop) for compression,
but LEARN the embeddings end-to-end instead of random.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Tuple
import numpy as np


class MemoryEncoder(nn.Module):
    """Encodes code snippets into memory embeddings.

    Each snippet becomes a single memory vector via mean pooling
    over learned token embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Project to memory space
        self.memory_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def encode_tokens(self, tokens: mx.array) -> mx.array:
        """Encode a sequence of tokens to a single memory vector.

        Args:
            tokens: (seq_len,) or (batch, seq_len) token IDs

        Returns:
            memory: (d_model,) or (batch, d_model) memory vector
        """
        # Get token embeddings
        emb = self.token_emb(tokens)  # (seq_len, d_model) or (batch, seq_len, d_model)

        # Mean pool over sequence
        if emb.ndim == 2:
            pooled = mx.mean(emb, axis=0)  # (d_model,)
        else:
            pooled = mx.mean(emb, axis=1)  # (batch, d_model)

        # Project and normalize
        memory = self.memory_proj(pooled)
        memory = self.ln(memory)

        return memory


class CrossAttention(nn.Module):
    """Cross-attention from query to memory bank.

    Query tokens attend over all memory slots to retrieve relevant context.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def __call__(
        self,
        query: mx.array,      # (batch, q_len, d_model)
        memory: mx.array,     # (batch, mem_len, d_model)
    ) -> Tuple[mx.array, mx.array]:
        """Cross-attention from query to memory.

        Returns:
            output: (batch, q_len, d_model)
            attn_weights: (batch, n_heads, q_len, mem_len)
        """
        B, Q, _ = query.shape
        _, M, _ = memory.shape

        # Project to Q, K, V
        q = self.W_q(query)   # (B, Q, d_model)
        k = self.W_k(memory)  # (B, M, d_model)
        v = self.W_v(memory)  # (B, M, d_model)

        # Reshape for multi-head attention
        q = q.reshape(B, Q, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B, H, Q, d_head)
        k = k.reshape(B, M, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B, H, M, d_head)
        v = v.reshape(B, M, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B, H, M, d_head)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, Q, M)
        attn = mx.softmax(scores, axis=-1)

        # Apply attention to values
        out = attn @ v  # (B, H, Q, d_head)

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, Q, self.d_model)
        out = self.W_o(out)

        return out, attn


class LTMBlock(nn.Module):
    """Single LTM block: self-attention + cross-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        # Self-attention (for query sequence)
        self.self_attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)

        # Cross-attention (query -> memory)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)

    def __call__(
        self,
        x: mx.array,          # (batch, seq_len, d_model) - query/decoder input
        memory: mx.array,     # (batch, mem_len, d_model) - memory bank
        mask: mx.array = None # Causal mask for self-attention
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Returns:
            output: (batch, seq_len, d_model)
            cross_attn_weights: (batch, n_heads, seq_len, mem_len)
        """
        # Self-attention with residual
        h = self.ln1(x)
        h = self.self_attn(h, h, h, mask=mask)[0]
        x = x + h

        # Cross-attention with residual
        h = self.ln2(x)
        cross_out, cross_attn = self.cross_attn(h, memory)
        x = x + cross_out

        # FFN with residual
        h = self.ln3(x)
        x = x + self.ffn(h)

        return x, cross_attn


class LTMModel(nn.Module):
    """Long-Term Memory Model for code generation.

    Architecture:
    1. Memory bank: Encoded code snippets (can be 10M+ tokens worth)
    2. Query encoder: Encodes the user query/prompt
    3. LTM blocks: Cross-attend to memory and generate

    Training:
    - Input: (memory_snippets, query, target_output)
    - Model learns to retrieve relevant snippets and generate correct output
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Memory encoder (for code snippets)
        self.memory_encoder = MemoryEncoder(vocab_size, d_model)

        # Query/decoder embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # LTM blocks
        self.blocks = [LTMBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def encode_memory(self, snippets: List[mx.array]) -> mx.array:
        """Encode a list of code snippets into memory bank.

        Args:
            snippets: List of (seq_len,) token arrays

        Returns:
            memory: (1, num_snippets, d_model) memory bank
        """
        memories = []
        for snippet in snippets:
            mem = self.memory_encoder.encode_tokens(snippet)
            memories.append(mem)

        # Stack into memory bank
        memory = mx.stack(memories, axis=0)  # (num_snippets, d_model)
        return memory[None, :, :]  # (1, num_snippets, d_model)

    def __call__(
        self,
        input_ids: mx.array,   # (batch, seq_len) - query + output tokens
        memory: mx.array,      # (batch, mem_len, d_model) - encoded memory
    ) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass.

        Args:
            input_ids: Token IDs for query/decoder input
            memory: Pre-encoded memory bank

        Returns:
            logits: (batch, seq_len, vocab_size)
            attn_weights: List of cross-attention weights per layer
        """
        B, L = input_ids.shape

        # Embed input
        h = self.token_emb(input_ids)
        positions = mx.arange(L)
        h = h + self.pos_emb(positions)

        # Causal mask for self-attention
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        # Expand memory for batch
        if memory.shape[0] == 1 and B > 1:
            memory = mx.broadcast_to(memory, (B, memory.shape[1], memory.shape[2]))

        # Apply LTM blocks
        all_attn = []
        for block in self.blocks:
            h, cross_attn = block(h, memory, mask)
            all_attn.append(cross_attn)

        # Output projection
        h = self.ln_f(h)
        logits = self.output(h)

        return logits, all_attn

    def generate(
        self,
        memory: mx.array,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
    ) -> List[int]:
        """Generate tokens given memory and prompt."""
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate to max length
            context = generated[-self.max_seq_len:]

            # Forward pass
            x = mx.array([context])
            logits, _ = self(x, memory)
            logits = logits[0, -1] / temperature

            # Sample
            probs = mx.softmax(logits)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            generated.append(int(next_token))

            # Stop on EOS
            if next_token == 3:
                break

        return generated

    def count_params(self) -> int:
        """Count parameters."""
        def _count(obj):
            if isinstance(obj, mx.array):
                return obj.size
            elif isinstance(obj, dict):
                return sum(_count(v) for v in obj.values())
            elif isinstance(obj, list):
                return sum(_count(v) for v in obj)
            return 0
        return _count(self.parameters())


def test_ltm_model():
    """Test the LTM model."""
    print("=" * 60)
    print("LTM Model Test")
    print("=" * 60)

    vocab_size = 10000
    model = LTMModel(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )

    print(f"\nModel parameters: {model.count_params():,}")

    # Create mock memory (5 code snippets)
    print("\nEncoding memory snippets...")
    snippets = [
        mx.array([100, 200, 300, 400, 500]),  # Mock snippet 1
        mx.array([150, 250, 350]),             # Mock snippet 2
        mx.array([200, 300, 400, 500, 600, 700]),  # Mock snippet 3
        mx.array([50, 100, 150, 200]),         # Mock snippet 4
        mx.array([300, 400, 500]),             # Mock snippet 5
    ]
    memory = model.encode_memory(snippets)
    print(f"  Memory shape: {memory.shape}")  # Should be (1, 5, d_model)

    # Forward pass
    print("\nForward pass...")
    batch_size = 2
    seq_len = 32
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    logits, attn_weights = model(input_ids, memory)
    mx.eval(logits)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Cross-attn shape: {attn_weights[0].shape}")

    # Generation
    print("\nGeneration test...")
    prompt = [2, 100, 200]  # BOS + tokens
    generated = model.generate(memory, prompt, max_new_tokens=10)
    print(f"  Prompt: {len(prompt)} tokens")
    print(f"  Generated: {len(generated)} tokens")

    print("\n✓ LTM model working!")


if __name__ == "__main__":
    test_ltm_model()
