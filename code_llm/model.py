"""Small Transformer Model for Code Generation.

A minimal transformer that can:
1. Learn code patterns from a codebase
2. Generate code using in-context examples
3. Make targeted edits based on instructions

Uses our hybrid tokenizer for efficient long-context handling.
"""

import numpy as np
from typing import Optional, List, Tuple
import pickle


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class Attention:
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Initialize weights
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (seq_len, seq_len) causal mask

        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q  # (batch, seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        # Now: (batch, n_heads, seq_len, d_head)

        # Attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        # (batch, n_heads, seq_len, seq_len)

        # Apply causal mask
        if mask is not None:
            scores = scores + mask[np.newaxis, np.newaxis, :, :]

        # Softmax
        attn = softmax(scores, axis=-1)

        # Apply attention to values
        out = attn @ V  # (batch, n_heads, seq_len, d_head)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

        # Output projection
        return out @ self.W_o


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        scale = 1.0 / np.sqrt(d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


class TransformerBlock:
    """Single transformer block with attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attention = Attention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: causal mask
        Returns:
            (batch, seq_len, d_model)
        """
        # Pre-norm attention
        h = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        x = x + self.attention.forward(h, mask)

        # Pre-norm FFN
        h = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        x = x + self.ffn.forward(h)

        return x


class SmallCodeLLM:
    """Small transformer for code generation.

    Architecture:
    - Embedding layer with positional encodings
    - N transformer blocks
    - Output projection to vocabulary
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 2048
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Token embeddings
        scale = 1.0 / np.sqrt(d_model)
        self.token_embeddings = np.random.randn(vocab_size, d_model) * scale

        # Positional encodings (learned)
        self.pos_embeddings = np.random.randn(max_seq_len, d_model) * scale

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.ln_f_gamma = np.ones(d_model)
        self.ln_f_beta = np.zeros(d_model)

        # Output projection (tied with embeddings)
        self.output_proj = self.token_embeddings.T  # (d_model, vocab_size)

        # Create causal mask
        self.causal_mask = np.triu(
            np.ones((max_seq_len, max_seq_len)) * -1e9, k=1
        )

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape

        # Get embeddings
        x = self.token_embeddings[token_ids]  # (batch, seq_len, d_model)
        x = x + self.pos_embeddings[:seq_len]  # Add positional

        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # Apply transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer norm
        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)

        # Project to vocabulary
        logits = x @ self.output_proj  # (batch, seq_len, vocab_size)

        return logits

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> List[int]:
        """Generate tokens autoregressively.

        Args:
            prompt_ids: List of token IDs for the prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            List of generated token IDs (including prompt)
        """
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = generated[-self.max_seq_len:]

            # Get logits for last position
            x = np.array([context])
            logits = self.forward(x)[0, -1]  # (vocab_size,)

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            top_k_indices = np.argsort(logits)[-top_k:]
            top_k_logits = logits[top_k_indices]

            # Convert to probabilities
            probs = softmax(top_k_logits)

            # Sample
            idx = np.random.choice(len(top_k_indices), p=probs)
            next_token = top_k_indices[idx]

            generated.append(int(next_token))

            # Stop on EOS
            if next_token == 3:  # <EOS>
                break

        return generated

    def count_params(self) -> int:
        """Count total parameters."""
        total = 0

        # Embeddings
        total += self.token_embeddings.size
        total += self.pos_embeddings.size

        # Transformer blocks
        for block in self.blocks:
            # Attention
            total += block.attention.W_q.size
            total += block.attention.W_k.size
            total += block.attention.W_v.size
            total += block.attention.W_o.size
            # FFN
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            # Layer norms
            total += block.ln1_gamma.size + block.ln1_beta.size
            total += block.ln2_gamma.size + block.ln2_beta.size

        # Final layer norm
        total += self.ln_f_gamma.size + self.ln_f_beta.size

        return total

    def save(self, path: str):
        """Save model weights."""
        state = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'token_embeddings': self.token_embeddings,
            'pos_embeddings': self.pos_embeddings,
            'blocks': [
                {
                    'attention': {
                        'W_q': b.attention.W_q,
                        'W_k': b.attention.W_k,
                        'W_v': b.attention.W_v,
                        'W_o': b.attention.W_o,
                    },
                    'ffn': {
                        'W1': b.ffn.W1,
                        'b1': b.ffn.b1,
                        'W2': b.ffn.W2,
                        'b2': b.ffn.b2,
                    },
                    'ln1_gamma': b.ln1_gamma,
                    'ln1_beta': b.ln1_beta,
                    'ln2_gamma': b.ln2_gamma,
                    'ln2_beta': b.ln2_beta,
                }
                for b in self.blocks
            ],
            'ln_f_gamma': self.ln_f_gamma,
            'ln_f_beta': self.ln_f_beta,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'SmallCodeLLM':
        """Load model weights."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        model = cls(
            vocab_size=state['vocab_size'],
            d_model=state['d_model'],
            n_layers=state['n_layers'],
            n_heads=state['n_heads'],
            d_ff=state['d_ff'],
            max_seq_len=state['max_seq_len'],
        )

        model.token_embeddings = state['token_embeddings']
        model.pos_embeddings = state['pos_embeddings']
        model.output_proj = model.token_embeddings.T

        for i, block_state in enumerate(state['blocks']):
            model.blocks[i].attention.W_q = block_state['attention']['W_q']
            model.blocks[i].attention.W_k = block_state['attention']['W_k']
            model.blocks[i].attention.W_v = block_state['attention']['W_v']
            model.blocks[i].attention.W_o = block_state['attention']['W_o']
            model.blocks[i].ffn.W1 = block_state['ffn']['W1']
            model.blocks[i].ffn.b1 = block_state['ffn']['b1']
            model.blocks[i].ffn.W2 = block_state['ffn']['W2']
            model.blocks[i].ffn.b2 = block_state['ffn']['b2']
            model.blocks[i].ln1_gamma = block_state['ln1_gamma']
            model.blocks[i].ln1_beta = block_state['ln1_beta']
            model.blocks[i].ln2_gamma = block_state['ln2_gamma']
            model.blocks[i].ln2_beta = block_state['ln2_beta']

        model.ln_f_gamma = state['ln_f_gamma']
        model.ln_f_beta = state['ln_f_beta']

        return model


def demo():
    """Demonstrate the model."""
    print("=" * 60)
    print("SMALL CODE LLM DEMO")
    print("=" * 60)

    # Create model
    vocab_size = 10000
    model = SmallCodeLLM(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=1024,
        max_seq_len=512
    )

    print(f"\nModel configuration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dim: {model.d_model}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Attention heads: {model.n_heads}")
    print(f"  FFN dim: {model.d_ff}")
    print(f"  Max sequence length: {model.max_seq_len}")
    print(f"\n  Total parameters: {model.count_params():,}")
    print(f"  Model size: ~{model.count_params() * 4 / 1e6:.1f} MB (fp32)")

    # Test forward pass
    print("\n" + "-" * 60)
    print("Testing forward pass...")
    batch_size = 2
    seq_len = 128
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    import time
    start = time.time()
    logits = model.forward(x)
    elapsed = time.time() - start

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Time: {elapsed*1000:.1f} ms")

    # Test generation
    print("\n" + "-" * 60)
    print("Testing generation...")
    prompt = [2, 100, 200, 300]  # <BOS> + some tokens
    start = time.time()
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    elapsed = time.time() - start

    print(f"  Prompt length: {len(prompt)}")
    print(f"  Generated length: {len(generated)}")
    print(f"  Time: {elapsed*1000:.1f} ms")
    print(f"  Tokens/sec: {len(generated) / elapsed:.1f}")


if __name__ == '__main__':
    demo()
