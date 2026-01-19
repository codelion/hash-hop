"""Quick 100K test - shorter training to get results faster."""

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
class Config:
    d_model: int = 512
    vocab_size: int = 128
    key_length: int = 4
    value_length: int = 4


class HashTable(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.key_encoder = nn.Sequential(
            nn.Linear(config.d_model * config.key_length, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size * config.value_length),
        )

    def encode_strings(self, strings: mx.array) -> mx.array:
        batch_size = strings.shape[0]
        char_embeds = self.char_embed(strings)
        flat = char_embeds.reshape(batch_size, -1)
        encoded = self.key_encoder(flat)
        norm = mx.sqrt(mx.sum(encoded ** 2, axis=-1, keepdims=True) + 1e-8)
        return encoded / norm

    def __call__(self, query, keys, values, temperature=0.001):
        batch_size = query.shape[0]
        num_pairs = keys.shape[1]

        query_embed = self.encode_strings(query)

        # Encode in chunks
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

        value_embeds_list = []
        for i in range(0, num_pairs, chunk_size):
            end = min(i + chunk_size, num_pairs)
            values_chunk = values[:, i:end, :]
            values_flat = values_chunk.reshape(batch_size * (end - i), -1)
            value_embeds_chunk = self.encode_strings(values_flat)
            value_embeds_chunk = value_embeds_chunk.reshape(batch_size, end - i, -1)
            value_embeds_list.append(value_embeds_chunk)
        value_embeds = mx.concatenate(value_embeds_list, axis=1)

        scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))
        scores = scores.squeeze(1)
        attention = mx.softmax(scores / temperature, axis=-1)

        retrieved = mx.matmul(attention[:, None, :], value_embeds)
        retrieved = retrieved.squeeze(1)

        output = self.value_decoder(retrieved)
        output = output.reshape(batch_size, self.config.value_length, self.config.vocab_size)

        return output, attention


def parse_context(context: str) -> List[Tuple[str, str]]:
    pairs = []
    pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
    for match in re.finditer(pattern, context):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def prepare_batch(samples, max_pairs, config):
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


def train_and_evaluate(context_size, max_steps=3000, batch_size=2):
    print(f"\n{'='*60}")
    print(f"Testing: {context_size:,} chars ({context_size // 10:,} pairs)")
    print(f"{'='*60}")

    config = Config()
    model = HashTable(config)
    max_pairs = context_size // 10 + 10

    eval_gen = MultiHopEval()
    optimizer = optim.AdamW(learning_rate=1e-4)

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

    print(f"Training for {max_steps} steps...")
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
            print(f"Step {step}: loss={float(loss):.4f}, time={elapsed:.0f}s")

        if step % 500 == 0:
            correct = 0
            total = 20
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
                    logits, _ = model(query_b, keys_b, values_b)
                    pred_chars = mx.argmax(logits[0], axis=-1)
                    pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
                    if pred_str == expected:
                        correct += 1
                    break
            accuracy = correct / total * 100
            print(f"  Eval: {accuracy:.0f}%")

    # Final eval
    print("\nFinal evaluation...")
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
            logits, _ = model(query_b, keys_b, values_b)
            pred_chars = mx.argmax(logits[0], axis=-1)
            pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())
            if pred_str == expected:
                correct += 1
            break

    accuracy = correct / total * 100
    print(f"\nFINAL: {context_size:,} chars = {accuracy:.0f}%")
    return accuracy


if __name__ == "__main__":
    results = {}
    for size in [100_000]:
        acc = train_and_evaluate(size, max_steps=3000, batch_size=2)
        results[size] = acc

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for ctx, acc in results.items():
        print(f"  {ctx:,} chars: {acc:.0f}%")
