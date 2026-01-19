"""Extreme Scale Hard Attention Hash Table for HashHop.

Goal: Solve HashHop at 100K-10M tokens by:
1. Larger embedding dimension (512) for better key discrimination
2. Hierarchical hashing to reduce the search space
3. More training steps
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Tuple
import re
import time
import gc

from hashhop import MultiHopEval


@dataclass
class ExtremeConfig:
    """Configuration for extreme scale."""
    d_model: int = 512  # Much larger embedding
    n_hash_buckets: int = 256  # Hierarchical: first hash to bucket, then search
    vocab_size: int = 128
    key_length: int = 4
    value_length: int = 4


class HierarchicalHashTable(nn.Module):
    """Two-level hash table for extreme scale.

    Level 1: Hash query to one of n_hash_buckets buckets (soft)
    Level 2: Hard attention within the bucket

    This reduces O(N) attention to O(N/buckets) per bucket.
    """

    def __init__(self, config: ExtremeConfig):
        super().__init__()
        self.config = config

        # Character embedding
        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Key encoder
        self.key_encoder = nn.Sequential(
            nn.Linear(config.d_model * config.key_length, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Bucket hash function (learned)
        self.bucket_proj = nn.Linear(config.d_model, config.n_hash_buckets)

        # Value decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size * config.value_length),
        )

    def encode_strings(self, strings: mx.array) -> mx.array:
        """Encode strings to normalized embeddings."""
        batch_size = strings.shape[0]
        char_embeds = self.char_embed(strings)
        flat = char_embeds.reshape(batch_size, -1)
        encoded = self.key_encoder(flat)
        # L2 normalize
        norm = mx.sqrt(mx.sum(encoded ** 2, axis=-1, keepdims=True) + 1e-8)
        return encoded / norm

    def __call__(
        self,
        query: mx.array,      # (batch, key_length)
        keys: mx.array,       # (batch, num_pairs, key_length)
        values: mx.array,     # (batch, num_pairs, value_length)
        temperature: float = 0.001,
    ) -> Tuple[mx.array, mx.array]:
        batch_size = query.shape[0]
        num_pairs = keys.shape[1]

        # Encode query
        query_embed = self.encode_strings(query)

        # Encode keys in chunks to save memory
        chunk_size = 500
        key_embeds_list = []
        for i in range(0, num_pairs, chunk_size):
            end = min(i + chunk_size, num_pairs)
            keys_chunk = keys[:, i:end, :]
            keys_flat = keys_chunk.reshape(batch_size * (end - i), -1)
            key_embeds_chunk = self.encode_strings(keys_flat)
            key_embeds_chunk = key_embeds_chunk.reshape(batch_size, end - i, -1)
            key_embeds_list.append(key_embeds_chunk)
        key_embeds = mx.concatenate(key_embeds_list, axis=1)

        # Encode values in chunks
        value_embeds_list = []
        for i in range(0, num_pairs, chunk_size):
            end = min(i + chunk_size, num_pairs)
            values_chunk = values[:, i:end, :]
            values_flat = values_chunk.reshape(batch_size * (end - i), -1)
            value_embeds_chunk = self.encode_strings(values_flat)
            value_embeds_chunk = value_embeds_chunk.reshape(batch_size, end - i, -1)
            value_embeds_list.append(value_embeds_chunk)
        value_embeds = mx.concatenate(value_embeds_list, axis=1)

        # Direct hard attention (no hierarchical for simplicity first)
        scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))
        scores = scores.squeeze(1)

        # Hard attention
        attention = mx.softmax(scores / temperature, axis=-1)

        # Retrieve
        retrieved = mx.matmul(attention[:, None, :], value_embeds)
        retrieved = retrieved.squeeze(1)

        # Decode
        output = self.value_decoder(retrieved)
        output = output.reshape(batch_size, self.config.value_length, self.config.vocab_size)

        return output, attention


def parse_context(context: str) -> List[Tuple[str, str]]:
    """Parse HashHop context into (key, value) pairs."""
    pairs = []
    pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
    for match in re.finditer(pattern, context):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def prepare_batch(
    samples: List[Tuple[str, str, str]],
    max_pairs: int,
    config: ExtremeConfig,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Prepare batch for training."""
    batch_queries = []
    batch_keys = []
    batch_values = []
    batch_targets = []

    for context, query, target in samples:
        pairs = parse_context(context)

        q_codes = [ord(c) for c in query[:config.key_length]]
        while len(q_codes) < config.key_length:
            q_codes.append(0)
        batch_queries.append(q_codes)

        k_codes = []
        v_codes = []
        for k, v in pairs[:max_pairs]:
            kc = [ord(c) for c in k[:config.key_length]]
            vc = [ord(c) for c in v[:config.value_length]]
            k_codes.append(kc)
            v_codes.append(vc)

        while len(k_codes) < max_pairs:
            k_codes.append([0] * config.key_length)
            v_codes.append([0] * config.value_length)

        batch_keys.append(k_codes)
        batch_values.append(v_codes)

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
    context_size: int,
    max_steps: int = 10000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    eval_every: int = 2000,
):
    """Train and evaluate."""
    print(f"\n{'='*70}")
    print(f"EXTREME SCALE: {context_size:,} chars ({context_size // 10:,} key-value pairs)")
    print(f"{'='*70}")

    config = ExtremeConfig()
    model = HierarchicalHashTable(config)

    max_pairs = context_size // 10 + 10

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
    print(f"Max pairs: {max_pairs:,}")
    print(f"Embedding dim: {config.d_model}")

    eval_gen = MultiHopEval()
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    def loss_fn(params, query, keys, values, targets):
        model.update(params)
        logits, _ = model(query, keys, values)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        target_log_probs = log_probs[batch_indices, targets_flat]
        return -target_log_probs.mean()

    loss_and_grad = mx.value_and_grad(loss_fn)

    print(f"\nTraining for {max_steps} steps with batch_size={batch_size}, lr={learning_rate}...")
    start_time = time.time()
    best_accuracy = 0

    for step in range(1, max_steps + 1):
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

        params = model.parameters()
        loss, grads = loss_and_grad(params, query, keys, values, targets)

        new_params = optimizer.apply_gradients(grads, params)
        model.update(new_params)
        mx.eval(model.parameters(), optimizer.state)

        if step % 200 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (max_steps - step) / steps_per_sec
            print(f"Step {step}: loss={float(loss):.4f}, time={elapsed:.0f}s, ETA={eta:.0f}s")

        if step % eval_every == 0:
            correct = 0
            total = 20  # Fewer for speed

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
                    logits, attn = model(query_b, keys_b, values_b)
                    pred_chars = mx.argmax(logits[0], axis=-1)
                    pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())

                    max_attn = float(mx.max(attn[0]).item())

                    if pred_str == expected:
                        correct += 1
                    break

            accuracy = correct / total * 100
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            print(f"  *** Eval accuracy: {accuracy:.1f}% (best: {best_accuracy:.1f}%), max_attn: {max_attn:.4f}")

    # Final evaluation
    print("\nFinal evaluation on 50 samples...")
    correct = 0
    total = 50

    for i in range(total):
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
            logits, _ = model(query_b, keys_b, values_b)
            pred_chars = mx.argmax(logits[0], axis=-1)
            pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
            if pred_str == expected:
                correct += 1
            break

    accuracy = correct / total * 100
    print(f"\nFINAL ACCURACY at {context_size:,} chars: {accuracy:.1f}%")

    del model, optimizer
    gc.collect()

    return accuracy


if __name__ == "__main__":
    results = {}

    # Focus on the challenging sizes
    test_sizes = [
        50_000,     # 5K pairs - Previous failure point
        100_000,    # 10K pairs
    ]

    for context_size in test_sizes:
        # More training for larger scales
        if context_size <= 50_000:
            max_steps = 10000
            batch_size = 4
        else:
            max_steps = 15000
            batch_size = 2

        acc = train_and_evaluate(
            context_size=context_size,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=1e-4,
            eval_every=2000,
        )
        results[context_size] = acc

    print("\n" + "="*70)
    print("EXTREME SCALE SUMMARY")
    print("="*70)
    for ctx, acc in sorted(results.items()):
        pairs = ctx // 10
        print(f"  {ctx:>10,} chars ({pairs:>6,} pairs): {acc:.1f}%")
