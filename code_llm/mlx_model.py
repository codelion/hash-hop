"""Small Transformer for Python Code Generation using MLX.

This uses Apple's MLX framework for proper autograd on Apple Silicon.
Key design: use our custom tokenizer where identifiers = single tokens.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple
import json


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        # Project to Q, K, V
        qkv = self.W_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Softmax and apply to values
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v

        # Reshape and project output
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SmallCodeLLM(nn.Module):
    """Small transformer for Python code generation.

    Architecture similar to GPT-2 but smaller:
    - Learned positional embeddings
    - Pre-norm transformer blocks
    - Tied input/output embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection (tied with token embeddings)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.output.weight = self.token_emb.weight

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, L = x.shape

        # Get embeddings
        h = self.token_emb(x)
        positions = mx.arange(L)
        h = h + self.pos_emb(positions)

        # Create causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, mask)

        # Final layer norm and output projection
        h = self.ln_f(h)
        logits = self.output(h)

        return logits

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> List[int]:
        """Generate tokens autoregressively."""
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = generated[-self.max_seq_len:]

            # Get logits for last position
            x = mx.array([context])
            logits = self(x)[0, -1]

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_idx = mx.argpartition(-logits, top_k)[:top_k]
                top_k_logits = logits[top_k_idx]

                # Softmax over top-k
                probs = mx.softmax(top_k_logits)

                # Sample
                idx = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = top_k_idx[idx]
            else:
                probs = mx.softmax(logits)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))

            generated.append(int(next_token))

            # Stop on EOS (token 3)
            if next_token == 3:
                break

        return generated

    def count_params(self) -> int:
        """Count total trainable parameters."""
        def _count(obj):
            if isinstance(obj, mx.array):
                return obj.size
            elif isinstance(obj, dict):
                return sum(_count(v) for v in obj.values())
            elif isinstance(obj, list):
                return sum(_count(v) for v in obj)
            return 0
        return _count(self.parameters())


def create_model(
    vocab_size: int,
    size: str = "small"
) -> SmallCodeLLM:
    """Create model with predefined size configurations.

    Sizes:
    - tiny: ~2M params (for quick testing)
    - small: ~10M params
    - medium: ~30M params
    """
    configs = {
        "tiny": {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512},
        "small": {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024},
        "medium": {"d_model": 512, "n_layers": 8, "n_heads": 8, "d_ff": 2048},
    }

    config = configs.get(size, configs["small"])
    return SmallCodeLLM(vocab_size=vocab_size, **config)


def test_model():
    """Quick test of the model."""
    print("=" * 60)
    print("MLX Code LLM Test")
    print("=" * 60)

    vocab_size = 10000
    model = create_model(vocab_size, "small")

    print(f"\nModel config:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_layers: {model.n_layers}")
    print(f"  Parameters: {model.count_params():,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nTesting forward pass...")
    print(f"  Input shape: {x.shape}")

    import time
    start = time.time()
    logits = model(x)
    mx.eval(logits)  # Force computation
    elapsed = time.time() - start

    print(f"  Output shape: {logits.shape}")
    print(f"  Time: {elapsed*1000:.1f}ms")

    # Test generation
    print(f"\nTesting generation...")
    prompt = [2, 100, 200, 300]  # BOS + some tokens
    start = time.time()
    generated = model.generate(prompt, max_new_tokens=20)
    elapsed = time.time() - start

    print(f"  Prompt: {len(prompt)} tokens")
    print(f"  Generated: {len(generated)} tokens")
    print(f"  Time: {elapsed*1000:.1f}ms")

    print("\n✓ Model working correctly!")


if __name__ == "__main__":
    test_model()
