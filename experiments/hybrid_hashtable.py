"""Hybrid Neural-Symbolic Hash Table for HashHop.

Key insight: Use symbolic parsing to extract structure, but neural networks
for the lookup operation. This tests whether the issue is:
1. Learning to parse (symbolic helps)
2. Learning to lookup (neural should be able to do this)

Approach:
- Symbolic: Parse context into (key, value) pairs
- Neural: Learn to match query to keys using embeddings + attention
- Neural: Learn to decode value from attention-weighted values
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import time

from hashhop import MultiHopEval


@dataclass
class HybridConfig:
    """Configuration for hybrid hash table."""
    d_model: int = 64  # Embedding dimension
    n_layers: int = 2  # Number of MLP layers
    vocab_size: int = 128  # ASCII characters
    key_length: int = 4
    value_length: int = 4


class LearnedHashTable(nn.Module):
    """A hash table where lookup is learned via neural attention."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        # Character embedding
        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Key encoder: 4 chars -> embedding
        self.key_encoder = nn.Sequential(
            nn.Linear(config.d_model * config.key_length, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )

        # Value decoder: embedding -> 4 char logits
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.vocab_size * config.value_length),
        )

    def encode_strings(self, strings: mx.array) -> mx.array:
        """
        Encode strings to embeddings.

        Args:
            strings: (batch, str_len) character codes

        Returns:
            embeddings: (batch, d_model)
        """
        batch_size = strings.shape[0]
        # Embed characters
        char_embeds = self.char_embed(strings)  # (batch, str_len, d_model)
        # Flatten and encode
        flat = char_embeds.reshape(batch_size, -1)
        return self.key_encoder(flat)

    def __call__(
        self,
        query: mx.array,      # (batch, key_length) query char codes
        keys: mx.array,       # (batch, num_pairs, key_length) key char codes
        values: mx.array,     # (batch, num_pairs, value_length) value char codes
        temperature: float = 0.1,
    ) -> mx.array:
        """
        Lookup query in the hash table.

        Args:
            query: Query key character codes
            keys: All keys in the table
            values: All values in the table
            temperature: Softmax temperature (lower = sharper)

        Returns:
            output_logits: (batch, value_length, vocab_size)
        """
        batch_size = query.shape[0]
        num_pairs = keys.shape[1]

        # Encode query
        query_embed = self.encode_strings(query)  # (batch, d_model)

        # Encode all keys
        keys_flat = keys.reshape(batch_size * num_pairs, -1)
        key_embeds = self.encode_strings(keys_flat)  # (batch * num_pairs, d_model)
        key_embeds = key_embeds.reshape(batch_size, num_pairs, -1)  # (batch, num_pairs, d_model)

        # Encode all values
        values_flat = values.reshape(batch_size * num_pairs, -1)
        value_embeds = self.encode_strings(values_flat)  # (batch * num_pairs, d_model)
        value_embeds = value_embeds.reshape(batch_size, num_pairs, -1)  # (batch, num_pairs, d_model)

        # Compute attention scores
        scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))  # (batch, 1, num_pairs)
        scores = scores / (self.config.d_model ** 0.5)
        scores = scores.squeeze(1)  # (batch, num_pairs)

        # Apply temperature and softmax
        attention = mx.softmax(scores / temperature, axis=-1)  # (batch, num_pairs)

        # Retrieve values via attention
        retrieved = mx.matmul(attention[:, None, :], value_embeds)  # (batch, 1, d_model)
        retrieved = retrieved.squeeze(1)  # (batch, d_model)

        # Decode to output
        output = self.value_decoder(retrieved)  # (batch, vocab_size * value_length)
        output = output.reshape(batch_size, self.config.value_length, self.config.vocab_size)

        return output, attention


def parse_context(context: str) -> List[Tuple[str, str]]:
    """Parse HashHop context into (key, value) pairs using symbolic regex."""
    pairs = []
    pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
    for match in re.finditer(pattern, context):
        key = match.group(1)
        value = match.group(2)
        pairs.append((key, value))
    return pairs


def prepare_batch(
    samples: List[Tuple[str, str, str]],  # (context, query, target)
    max_pairs: int = 100,
    config: HybridConfig = None,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Prepare a batch for training.

    Returns:
        query: (batch, key_length)
        keys: (batch, max_pairs, key_length)
        values: (batch, max_pairs, value_length)
        targets: (batch, value_length)
    """
    if config is None:
        config = HybridConfig()

    batch_queries = []
    batch_keys = []
    batch_values = []
    batch_targets = []

    for context, query, target in samples:
        # Parse context
        pairs = parse_context(context)

        # Prepare query
        q_codes = [ord(c) for c in query[:config.key_length]]
        while len(q_codes) < config.key_length:
            q_codes.append(0)
        batch_queries.append(q_codes)

        # Prepare keys and values
        k_codes = []
        v_codes = []
        for k, v in pairs[:max_pairs]:
            kc = [ord(c) for c in k[:config.key_length]]
            vc = [ord(c) for c in v[:config.value_length]]
            k_codes.append(kc)
            v_codes.append(vc)

        # Pad to max_pairs
        while len(k_codes) < max_pairs:
            k_codes.append([0] * config.key_length)
            v_codes.append([0] * config.value_length)

        batch_keys.append(k_codes)
        batch_values.append(v_codes)

        # Prepare target
        t_codes = [ord(c) for c in target[:config.value_length]]
        while len(t_codes) < config.value_length:
            t_codes.append(0)
        batch_targets.append(t_codes)

    return (
        mx.array(batch_queries, dtype=mx.int32),
        mx.array(batch_keys, dtype=mx.int32),
        mx.array(batch_values, dtype=mx.int32),
        mx.array(batch_targets, dtype=mx.int32),
    )


def train_and_evaluate(
    context_size: int = 500,
    max_steps: int = 3000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    eval_every: int = 500,
):
    """Train and evaluate the hybrid hash table."""
    print(f"\n{'='*60}")
    print(f"Training Hybrid (Neural Lookup + Symbolic Parse)")
    print(f"Context size: {context_size} chars")
    print(f"{'='*60}")

    config = HybridConfig()
    model = LearnedHashTable(config)

    # Estimate max pairs
    max_pairs = context_size // 10 + 10

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

    eval_gen = MultiHopEval()
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def loss_fn(params, query, keys, values, targets):
        model.update(params)
        logits, _ = model(query, keys, values)  # (batch, value_length, vocab_size)
        # Cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        target_log_probs = log_probs[batch_indices, targets_flat]
        return -target_log_probs.mean()

    loss_and_grad = mx.value_and_grad(loss_fn)

    print(f"\nTraining for {max_steps} steps...")
    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Generate batch
        samples = []
        for _ in range(batch_size):
            sample = eval_gen.make_one(
                n_chars_problem=context_size,
                num_queries=1,
                hops=1,
                hash_pair_str_length=4,
                chain_of_thought=False,
            )
            for q, t in sample.targets.items():
                samples.append((sample.prompt, q, t))
                break

        query, keys, values, targets = prepare_batch(samples, max_pairs, config)

        # Forward and backward
        params = model.parameters()
        loss, grads = loss_and_grad(params, query, keys, values, targets)

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
                    query_b, keys_b, values_b, _ = prepare_batch(
                        [(sample.prompt, q, expected)], max_pairs, config
                    )
                    logits, attention = model(query_b, keys_b, values_b)
                    pred_chars = mx.argmax(logits[0], axis=-1)
                    pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
                    if pred_str == expected:
                        correct += 1
                    break

            accuracy = correct / total * 100
            print(f"  Eval accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Final evaluation
    print("\nFinal evaluation on 200 samples...")
    correct = 0
    total = 200

    for _ in range(total):
        sample = eval_gen.make_one(
            n_chars_problem=context_size,
            num_queries=1,
            hops=1,
            hash_pair_str_length=4,
            chain_of_thought=False,
        )
        for q, expected in sample.targets.items():
            query_b, keys_b, values_b, _ = prepare_batch(
                [(sample.prompt, q, expected)], max_pairs, config
            )
            logits, attention = model(query_b, keys_b, values_b)
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
    results = {}
    for context_size in [200, 500, 1000, 2000]:
        acc = train_and_evaluate(
            context_size=context_size,
            max_steps=3000,
            batch_size=16,
            learning_rate=1e-3,
            eval_every=500,
        )
        results[context_size] = acc

    print("\n" + "="*60)
    print("SUMMARY: Hybrid Neural-Symbolic Hash Table")
    print("="*60)
    for ctx, acc in results.items():
        print(f"  {ctx:>5} chars: {acc:.1f}%")
