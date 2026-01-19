"""Fully Neural Hash Table for HashHop.

Unlike the previous implementation that used regex parsing, this one:
1. Uses a neural encoder to process the raw context string
2. Uses learned attention to find the relevant key-value pair
3. Uses learned decoding to produce the value

This tests whether neural networks can learn hash table lookup end-to-end.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

from hashhop import MultiHopEval


@dataclass
class NeuralConfig:
    """Configuration for neural hash table."""
    d_model: int = 128  # Embedding dimension
    n_heads: int = 4  # Attention heads
    n_layers: int = 2  # Transformer layers
    max_seq_len: int = 2048  # Max context length
    vocab_size: int = 128  # ASCII characters
    output_len: int = 4  # Output value length


class CharTransformer(nn.Module):
    """Character-level transformer for hash table lookup."""

    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config

        # Character embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer encoder layers
        self.layers = [
            TransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ]

        # Query projection (for the lookup key)
        self.query_proj = nn.Linear(config.d_model * 4, config.d_model)

        # Output head
        self.output_head = nn.Linear(config.d_model, config.vocab_size * config.output_len)

    def __call__(
        self,
        context: mx.array,  # (batch, seq_len) character codes
        query: mx.array,    # (batch, 4) query key character codes
    ) -> mx.array:
        """
        Args:
            context: Full HashHop context as character codes
            query: 4-character query key

        Returns:
            output_logits: (batch, output_len, vocab_size)
        """
        batch_size, seq_len = context.shape

        # Embed context
        positions = mx.arange(seq_len)
        x = self.embed(context) + self.pos_embed(positions)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Embed query and project
        query_embed = self.embed(query)  # (batch, 4, d_model)
        query_flat = query_embed.reshape(batch_size, -1)  # (batch, 4 * d_model)
        query_vec = self.query_proj(query_flat)  # (batch, d_model)

        # Compute attention from query to context
        scores = mx.matmul(query_vec[:, None, :], x.transpose(0, 2, 1))  # (batch, 1, seq_len)
        scores = scores / (self.config.d_model ** 0.5)
        attention = mx.softmax(scores, axis=-1)

        # Retrieve context based on attention
        retrieved = mx.matmul(attention, x)  # (batch, 1, d_model)
        retrieved = retrieved.squeeze(1)  # (batch, d_model)

        # Output projection
        output = self.output_head(retrieved)  # (batch, vocab_size * output_len)
        output = output.reshape(batch_size, self.config.output_len, self.config.vocab_size)

        return output


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with residual
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def prepare_sample(
    context: str,
    query: str,
    target: str,
    max_len: int = 2048,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Prepare a single sample for training."""
    # Convert context to character codes
    context_codes = [ord(c) for c in context[:max_len]]
    # Pad if needed
    while len(context_codes) < max_len:
        context_codes.append(0)

    # Convert query to character codes
    query_codes = [ord(c) for c in query[:4]]
    while len(query_codes) < 4:
        query_codes.append(0)

    # Convert target to character codes
    target_codes = [ord(c) for c in target[:4]]
    while len(target_codes) < 4:
        target_codes.append(0)

    return (
        mx.array([context_codes], dtype=mx.int32),
        mx.array([query_codes], dtype=mx.int32),
        mx.array([target_codes], dtype=mx.int32),
    )


def train_and_evaluate(
    context_size: int = 500,
    max_steps: int = 5000,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    eval_every: int = 500,
):
    """Train and evaluate the neural hash table."""
    print(f"\n{'='*60}")
    print(f"Training Neural Hash Table on {context_size}-char contexts")
    print(f"{'='*60}")

    # Create model
    config = NeuralConfig(max_seq_len=context_size + 100)
    model = CharTransformer(config)

    # Count parameters
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, dict):
                total += count_params(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        total += count_params(item)
        return total

    num_params = count_params(model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create data generator
    eval_gen = MultiHopEval()

    # Create optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def loss_fn(params, context, query, target):
        model.update(params)
        logits = model(context, query)  # (batch, 4, vocab_size)
        # Cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        target_log_probs = log_probs[batch_indices, target_flat]
        return -target_log_probs.mean()

    loss_and_grad = mx.value_and_grad(loss_fn)

    print(f"\nTraining for {max_steps} steps with batch_size={batch_size}...")
    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Generate batch
        contexts = []
        queries = []
        targets = []

        for _ in range(batch_size):
            sample = eval_gen.make_one(
                n_chars_problem=context_size,
                num_queries=1,
                hops=1,
                hash_pair_str_length=4,
                chain_of_thought=False,
            )
            # Get first query/target pair
            for q, t in sample.targets.items():
                ctx_codes = [ord(c) for c in sample.prompt[:config.max_seq_len]]
                while len(ctx_codes) < config.max_seq_len:
                    ctx_codes.append(0)
                contexts.append(ctx_codes)

                q_codes = [ord(c) for c in q[:4]]
                while len(q_codes) < 4:
                    q_codes.append(0)
                queries.append(q_codes)

                t_codes = [ord(c) for c in t[:4]]
                while len(t_codes) < 4:
                    t_codes.append(0)
                targets.append(t_codes)
                break

        context_batch = mx.array(contexts, dtype=mx.int32)
        query_batch = mx.array(queries, dtype=mx.int32)
        target_batch = mx.array(targets, dtype=mx.int32)

        # Forward and backward
        params = model.parameters()
        loss, grads = loss_and_grad(params, context_batch, query_batch, target_batch)

        # Update
        new_params = optimizer.apply_gradients(grads, params)
        model.update(new_params)
        mx.eval(model.parameters(), optimizer.state)

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: loss={float(loss):.4f}, time={elapsed:.1f}s")

        # Evaluate
        if step % eval_every == 0:
            correct = 0
            total = 50

            for _ in range(total):
                sample = eval_gen.make_one(
                    n_chars_problem=context_size,
                    num_queries=1,
                    hops=1,
                    hash_pair_str_length=4,
                    chain_of_thought=False,
                )
                for q, expected in sample.targets.items():
                    ctx, qry, _ = prepare_sample(sample.prompt, q, expected, config.max_seq_len)
                    logits = model(ctx, qry)
                    pred_chars = mx.argmax(logits[0], axis=-1)
                    pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
                    if pred_str == expected:
                        correct += 1
                    break

            accuracy = correct / total * 100
            print(f"  Eval accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Final evaluation
    print("\nFinal evaluation on 100 samples...")
    correct = 0
    total = 100

    for _ in range(total):
        sample = eval_gen.make_one(
            n_chars_problem=context_size,
            num_queries=1,
            hops=1,
            hash_pair_str_length=4,
            chain_of_thought=False,
        )
        for q, expected in sample.targets.items():
            ctx, qry, _ = prepare_sample(sample.prompt, q, expected, config.max_seq_len)
            logits = model(ctx, qry)
            pred_chars = mx.argmax(logits[0], axis=-1)
            pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
            if pred_str == expected:
                correct += 1
            break

    accuracy = correct / total * 100
    print(f"Final accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    # Test on various context sizes
    for context_size in [200, 500]:
        train_and_evaluate(
            context_size=context_size,
            max_steps=3000,
            batch_size=8,
            learning_rate=3e-4,
            eval_every=500,
        )
