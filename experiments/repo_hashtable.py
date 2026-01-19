"""RePo-inspired Hash Table for HashHop.

Key insight from RePo (Sakana AI): Instead of using linear positional encoding,
learn semantic-based positions that bring related tokens closer in positional space.

For HashHop, this means:
1. Learn position embeddings based on key content (not index)
2. Keys with similar content get similar positions
3. Query's position should be close to its matching key's position
4. This reduces the effective "distance" for attention

The core idea: if query "ABCD" and key "ABCD" have the same learned position,
attention will naturally focus on the matching key regardless of how many
other keys exist in the context.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Tuple
import re
import time
import math

from hashhop import MultiHopEval


@dataclass
class RePoConfig:
    """Configuration for RePo-inspired hash table."""
    d_model: int = 256
    d_pos: int = 64  # Position embedding dimension
    vocab_size: int = 128
    key_length: int = 4
    value_length: int = 4
    n_pos_heads: int = 4  # Multiple position heads for different aspects


class PositionPredictor(nn.Module):
    """Predicts semantic-based position from token content.

    Instead of linear positions (0, 1, 2, ..., N), predicts a
    continuous position based on the content of each key.
    """

    def __init__(self, config: RePoConfig):
        super().__init__()
        self.config = config

        # MLP to predict position from key embedding
        self.pos_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_pos),
            nn.GELU(),
            nn.Linear(config.d_pos, config.n_pos_heads),  # One position per head
        )

    def __call__(self, key_embeds: mx.array) -> mx.array:
        """
        Args:
            key_embeds: (batch, num_keys, d_model)

        Returns:
            positions: (batch, num_keys, n_pos_heads) - continuous positions
        """
        return self.pos_mlp(key_embeds)


class RePoHashTable(nn.Module):
    """Hash table with learned semantic positions (RePo-style).

    The key innovation: attention scores are modified by positional distance
    in a learned semantic space. If query and key have similar content,
    they get similar positions, making attention sharper.
    """

    def __init__(self, config: RePoConfig):
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

        # Position predictor (RePo's core idea)
        self.pos_predictor = PositionPredictor(config)

        # Value decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size * config.value_length),
        )

        # Position-based attention scaling
        self.pos_scale = 10.0  # How strongly position affects attention

    def encode_strings(self, strings: mx.array) -> mx.array:
        """Encode strings to embeddings."""
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
        temperature: float = 0.01,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass with RePo-style positional attention.

        Returns:
            output_logits: (batch, value_length, vocab_size)
            attention: (batch, num_pairs)
            pos_distances: (batch, num_pairs) - for debugging
        """
        batch_size = query.shape[0]
        num_pairs = keys.shape[1]

        # Encode query
        query_embed = self.encode_strings(query)  # (batch, d_model)

        # Encode all keys in chunks
        chunk_size = 500
        key_embeds_list = []
        for i in range(0, num_pairs, chunk_size):
            end = min(i + chunk_size, num_pairs)
            keys_chunk = keys[:, i:end, :]
            keys_flat = keys_chunk.reshape(batch_size * (end - i), -1)
            key_embeds_chunk = self.encode_strings(keys_flat)
            key_embeds_chunk = key_embeds_chunk.reshape(batch_size, end - i, -1)
            key_embeds_list.append(key_embeds_chunk)
        key_embeds = mx.concatenate(key_embeds_list, axis=1)  # (batch, num_pairs, d_model)

        # Encode values
        value_embeds_list = []
        for i in range(0, num_pairs, chunk_size):
            end = min(i + chunk_size, num_pairs)
            values_chunk = values[:, i:end, :]
            values_flat = values_chunk.reshape(batch_size * (end - i), -1)
            value_embeds_chunk = self.encode_strings(values_flat)
            value_embeds_chunk = value_embeds_chunk.reshape(batch_size, end - i, -1)
            value_embeds_list.append(value_embeds_chunk)
        value_embeds = mx.concatenate(value_embeds_list, axis=1)

        # === RePo Innovation: Predict semantic positions ===

        # Get position for query
        query_pos = self.pos_predictor(query_embed[:, None, :])  # (batch, 1, n_heads)

        # Get positions for all keys
        key_pos = self.pos_predictor(key_embeds)  # (batch, num_pairs, n_heads)

        # Compute positional distance (L2 in position space)
        pos_diff = query_pos - key_pos  # (batch, num_pairs, n_heads)
        pos_distance = mx.sqrt(mx.sum(pos_diff ** 2, axis=-1) + 1e-8)  # (batch, num_pairs)

        # === Compute attention with position-aware scoring ===

        # Content-based similarity (cosine, since embeddings are normalized)
        content_scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))
        content_scores = content_scores.squeeze(1)  # (batch, num_pairs)

        # Position-based bonus: closer positions get higher scores
        # Similar content -> similar positions -> smaller distance -> higher score
        pos_bonus = -self.pos_scale * pos_distance  # Negative distance = bonus

        # Combined score
        combined_scores = content_scores + pos_bonus

        # Hard attention with low temperature
        attention = mx.softmax(combined_scores / temperature, axis=-1)

        # Retrieve via attention
        retrieved = mx.matmul(attention[:, None, :], value_embeds)
        retrieved = retrieved.squeeze(1)

        # Decode
        output = self.value_decoder(retrieved)
        output = output.reshape(batch_size, self.config.value_length, self.config.vocab_size)

        return output, attention, pos_distance


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


def train_and_evaluate(context_size, max_steps=5000, batch_size=8, eval_every=1000):
    print(f"\n{'='*60}")
    print(f"RePo Hash Table: {context_size:,} chars ({context_size // 10:,} pairs)")
    print(f"{'='*60}")

    config = RePoConfig()
    model = RePoHashTable(config)
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
    optimizer = optim.AdamW(learning_rate=3e-4)

    def loss_fn(params, query, keys, values, targets):
        model.update(params)
        logits, attention, pos_dist = model(query, keys, values)

        # Main loss: cross-entropy on value prediction
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        batch_indices = mx.arange(logits_flat.shape[0])
        target_log_probs = log_probs[batch_indices, targets_flat]
        ce_loss = -target_log_probs.mean()

        # Auxiliary loss: encourage attention entropy to be low (sharp attention)
        # This helps the position predictor learn to separate keys
        entropy = -mx.sum(attention * mx.log(attention + 1e-10), axis=-1).mean()
        entropy_loss = 0.1 * entropy  # Small weight

        return ce_loss + entropy_loss

    loss_and_grad = mx.value_and_grad(loss_fn)

    print(f"Training for {max_steps} steps...")
    start_time = time.time()
    best_acc = 0

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

        if step % eval_every == 0:
            correct = 0
            total = 30
            avg_max_attn = 0
            avg_min_dist = 0

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
                    logits, attn, pos_dist = model(query_b, keys_b, values_b)
                    pred_chars = mx.argmax(logits[0], axis=-1)
                    pred_str = "".join(chr(int(c)) for c in pred_chars.tolist())

                    max_attn = float(mx.max(attn[0]).item())
                    min_dist = float(mx.min(pos_dist[0]).item())
                    avg_max_attn += max_attn
                    avg_min_dist += min_dist

                    if pred_str == expected:
                        correct += 1
                    break

            accuracy = correct / total * 100
            avg_max_attn /= total
            avg_min_dist /= total
            if accuracy > best_acc:
                best_acc = accuracy
            print(f"  Eval: {accuracy:.0f}% (best: {best_acc:.0f}%), max_attn: {avg_max_attn:.4f}, min_pos_dist: {avg_min_dist:.4f}")

    # Final evaluation
    print("\nFinal evaluation...")
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
            query_b, keys_b, values_b, _ = prepare_batch(
                [(sample.prompt, q, expected)], max_pairs, config
            )
            logits, _, _ = model(query_b, keys_b, values_b)
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

    # Test at various scales
    for size in [2000, 5000, 10000, 50000]:
        if size <= 5000:
            max_steps = 5000
            batch_size = 8
        elif size <= 10000:
            max_steps = 5000
            batch_size = 4
        else:
            max_steps = 5000
            batch_size = 2

        acc = train_and_evaluate(size, max_steps=max_steps, batch_size=batch_size)
        results[size] = acc

    print("\n" + "="*60)
    print("RePo Hash Table RESULTS")
    print("="*60)
    for ctx, acc in sorted(results.items()):
        print(f"  {ctx:>10,} chars ({ctx//10:>6,} pairs): {acc:.0f}%")
