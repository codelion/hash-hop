"""Test Hard Attention Hash Table at extreme scales.

Goal: See if we can solve HashHop at 10M+ tokens using hard attention.

Key insight from previous experiments:
- Hard attention (temp=0.001) scales to 2K chars with ~90% accuracy
- Now testing: 5K, 10K, 50K, 100K, 500K, 1M+ chars
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
class ScalableConfig:
    """Configuration optimized for scale."""
    d_model: int = 256  # Larger embedding for better discrimination at scale
    vocab_size: int = 128
    key_length: int = 4
    value_length: int = 4


class ScalableHashTable(nn.Module):
    """Hash table optimized for extreme scale."""

    def __init__(self, config: ScalableConfig):
        super().__init__()
        self.config = config

        # Character embedding
        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Key encoder with deeper network for better representations
        self.key_encoder = nn.Sequential(
            nn.Linear(config.d_model * config.key_length, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Value decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.vocab_size * config.value_length),
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
        query: mx.array,
        keys: mx.array,
        values: mx.array,
        temperature: float = 0.001,
    ) -> Tuple[mx.array, mx.array]:
        batch_size = query.shape[0]
        num_pairs = keys.shape[1]

        # Encode query
        query_embed = self.encode_strings(query)

        # Encode keys in chunks to save memory for large num_pairs
        chunk_size = 1000
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

        # Compute attention scores
        scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))
        scores = scores.squeeze(1)

        # Hard attention with very low temperature
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
    config: ScalableConfig,
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
    max_steps: int = 5000,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    eval_every: int = 1000,
):
    """Train and evaluate at a specific context size."""
    print(f"\n{'='*70}")
    print(f"SCALE TEST: {context_size:,} chars ({context_size // 10:,} key-value pairs)")
    print(f"{'='*70}")

    config = ScalableConfig()
    model = ScalableHashTable(config)

    # Estimate max pairs (roughly 10 chars per pair: "ABCD = EFGH\n")
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

    eval_gen = MultiHopEval()
    optimizer = optim.AdamW(learning_rate=learning_rate)

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

    print(f"\nTraining for {max_steps} steps with batch_size={batch_size}...")
    start_time = time.time()

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

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: loss={float(loss):.4f}, time={elapsed:.1f}s")

        if step % eval_every == 0:
            correct = 0
            total = 30  # Fewer samples for speed at large scales

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
            print(f"  Eval accuracy: {accuracy:.1f}% ({correct}/{total}), max_attn: {max_attn:.4f}")

    # Final evaluation
    print("\nFinal evaluation on 100 samples...")
    correct = 0
    total = 100

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

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{total}...")

    accuracy = correct / total * 100
    print(f"\nFINAL ACCURACY at {context_size:,} chars: {accuracy:.1f}% ({correct}/{total})")

    # Clean up
    del model, optimizer
    gc.collect()

    return accuracy


if __name__ == "__main__":
    results = {}

    # Start with moderate sizes to verify training works
    test_sizes = [
        5_000,      # 500 pairs
        10_000,     # 1K pairs
        50_000,     # 5K pairs
        100_000,    # 10K pairs
    ]

    for context_size in test_sizes:
        # Adjust training based on scale
        if context_size <= 10_000:
            max_steps = 5000
            batch_size = 8
        elif context_size <= 50_000:
            max_steps = 5000
            batch_size = 4
        else:
            max_steps = 5000
            batch_size = 2

        acc = train_and_evaluate(
            context_size=context_size,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=3e-4,
            eval_every=1000,
        )
        results[context_size] = acc

        # Stop if accuracy drops too low
        if acc < 50:
            print(f"\n⚠️ Accuracy dropped below 50% at {context_size:,} chars. Stopping.")
            break

    print("\n" + "="*70)
    print("SCALE TEST SUMMARY: Hard Attention Hash Table")
    print("="*70)
    for ctx, acc in sorted(results.items()):
        pairs = ctx // 10
        print(f"  {ctx:>10,} chars ({pairs:>6,} pairs): {acc:.1f}%")
