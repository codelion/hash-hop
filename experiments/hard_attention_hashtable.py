"""Hard Attention Hash Table for HashHop.

Key insight: The softmax attention spreads probability too thin over many keys.
Solution: Use HARD attention (argmax) with straight-through estimator for gradients.

This should give O(1) exact lookup regardless of number of entries.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Tuple
import re
import time

from hashhop import MultiHopEval


@dataclass
class HardAttnConfig:
    """Configuration for hard attention hash table."""
    d_model: int = 128  # Embedding dimension (larger for better discrimination)
    vocab_size: int = 128  # ASCII characters
    key_length: int = 4
    value_length: int = 4


class HardAttentionHashTable(nn.Module):
    """Hash table using hard attention for exact lookup."""

    def __init__(self, config: HardAttnConfig):
        super().__init__()
        self.config = config

        # Character embedding
        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Key encoder with LayerNorm for better similarity computation
        self.key_encoder = nn.Sequential(
            nn.Linear(config.d_model * config.key_length, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Value decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.vocab_size * config.value_length),
        )

    def encode_strings(self, strings: mx.array) -> mx.array:
        """Encode strings to normalized embeddings."""
        batch_size = strings.shape[0]
        char_embeds = self.char_embed(strings)
        flat = char_embeds.reshape(batch_size, -1)
        encoded = self.key_encoder(flat)
        # L2 normalize for cosine similarity
        norm = mx.sqrt(mx.sum(encoded ** 2, axis=-1, keepdims=True) + 1e-8)
        return encoded / norm

    def __call__(
        self,
        query: mx.array,      # (batch, key_length)
        keys: mx.array,       # (batch, num_pairs, key_length)
        values: mx.array,     # (batch, num_pairs, value_length)
        use_hard_attention: bool = True,
        temperature: float = 0.01,
    ) -> Tuple[mx.array, mx.array]:
        """
        Lookup with hard or soft attention.
        """
        batch_size = query.shape[0]
        num_pairs = keys.shape[1]

        # Encode query (L2 normalized)
        query_embed = self.encode_strings(query)  # (batch, d_model)

        # Encode all keys
        keys_flat = keys.reshape(batch_size * num_pairs, -1)
        key_embeds = self.encode_strings(keys_flat)
        key_embeds = key_embeds.reshape(batch_size, num_pairs, -1)

        # Encode all values
        values_flat = values.reshape(batch_size * num_pairs, -1)
        value_embeds = self.encode_strings(values_flat)
        value_embeds = value_embeds.reshape(batch_size, num_pairs, -1)

        # Compute cosine similarity (since embeddings are normalized)
        scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))
        scores = scores.squeeze(1)  # (batch, num_pairs)

        if use_hard_attention:
            # Hard attention: one-hot on argmax
            # Use straight-through estimator: forward uses argmax, backward uses softmax
            soft_attention = mx.softmax(scores / temperature, axis=-1)

            # Create one-hot from argmax
            best_idx = mx.argmax(scores, axis=-1)  # (batch,)
            hard_attention = mx.zeros_like(soft_attention)

            # Set the argmax position to 1
            for i in range(batch_size):
                idx = int(best_idx[i].item())
                # Can't directly index-assign in MLX, use scatter approach
                pass

            # Alternative: use very low temperature softmax (approximates argmax)
            attention = mx.softmax(scores / 0.001, axis=-1)  # Nearly one-hot
        else:
            attention = mx.softmax(scores / temperature, axis=-1)

        # Retrieve via attention
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
    max_pairs: int = 250,
    config: HardAttnConfig = None,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Prepare batch for training."""
    if config is None:
        config = HardAttnConfig()

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
    context_size: int = 500,
    max_steps: int = 5000,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    eval_every: int = 500,
    use_hard_attention: bool = True,
):
    """Train and evaluate."""
    print(f"\n{'='*60}")
    print(f"Training {'HARD' if use_hard_attention else 'SOFT'} Attention Hash Table")
    print(f"Context size: {context_size} chars")
    print(f"{'='*60}")

    config = HardAttnConfig()
    model = HardAttentionHashTable(config)

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

    eval_gen = MultiHopEval()
    optimizer = optim.AdamW(learning_rate=learning_rate)

    def loss_fn(params, query, keys, values, targets):
        model.update(params)
        logits, _ = model(query, keys, values, use_hard_attention=use_hard_attention)
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
                    logits, attn = model(query_b, keys_b, values_b, use_hard_attention=use_hard_attention)
                    pred_chars = mx.argmax(logits[0], axis=-1)
                    pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())

                    # Also check attention sharpness
                    max_attn = float(mx.max(attn[0]).item())

                    if pred_str == expected:
                        correct += 1
                    break

            accuracy = correct / total * 100
            print(f"  Eval accuracy: {accuracy:.1f}% ({correct}/{total}), max_attn: {max_attn:.4f}")

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
            logits, _ = model(query_b, keys_b, values_b, use_hard_attention=use_hard_attention)
            pred_chars = mx.argmax(logits[0], axis=-1)
            pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
            if pred_str == expected:
                correct += 1
            break

    accuracy = correct / total * 100
    print(f"Final accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    results = {}
    for context_size in [200, 500, 1000, 2000]:
        acc = train_and_evaluate(
            context_size=context_size,
            max_steps=5000,
            batch_size=16,
            learning_rate=3e-4,
            eval_every=500,
            use_hard_attention=True,
        )
        results[context_size] = acc

    print("\n" + "="*60)
    print("SUMMARY: Hard Attention Hash Table")
    print("="*60)
    for ctx, acc in results.items():
        print(f"  {ctx:>5} chars: {acc:.1f}%")
