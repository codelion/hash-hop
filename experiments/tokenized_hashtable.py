"""Tokenized Hash Table for HashHop.

Key insight from user: Instead of treating "ABCD" as 4 characters that need
to be matched exactly, treat each unique 4-char string as a SINGLE TOKEN.

This converts HashHop into MQAR (Multi-Query Associative Recall), which
transformers are proven to solve perfectly!

The difference:
- Character-level: Model must learn that "ABCD" == "ABCD" by comparing 4 chars
- Token-level: Model just needs to match token ID 42 == token ID 42

This is exactly what induction heads do in natural language!
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import List, Tuple, Dict
import re
import time

from hashhop import MultiHopEval


@dataclass
class TokenConfig:
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 1  # Single layer should be enough for MQAR
    max_vocab: int = 5000  # Max unique tokens (keys/values)


class TokenizedHashTable(nn.Module):
    """Hash table operating on token IDs instead of character strings.

    Each unique 4-char string gets a unique token ID.
    The model just needs to match token IDs - much easier!
    """

    def __init__(self, config: TokenConfig):
        super().__init__()
        self.config = config

        # Token embedding - each unique string gets an embedding
        self.token_embed = nn.Embedding(config.max_vocab, config.d_model)

        # Simple 2-layer transformer (like induction head circuit)
        # Layer 1: Encode key-value associations
        # Layer 2: Match query to keys and retrieve values
        self.layers = [
            TransformerLayer(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ]

        # Output projection back to token IDs
        self.output_proj = nn.Linear(config.d_model, config.max_vocab)

    def __call__(
        self,
        query_tokens: mx.array,    # (batch,) - single token ID per query
        key_tokens: mx.array,      # (batch, num_pairs) - token IDs
        value_tokens: mx.array,    # (batch, num_pairs) - token IDs
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            query_tokens: Token ID of the query key
            key_tokens: Token IDs of all keys in the table
            value_tokens: Token IDs of all values in the table

        Returns:
            logits: (batch, max_vocab) - predicted token ID distribution
            attention: (batch, num_pairs) - attention weights
        """
        batch_size = query_tokens.shape[0]
        num_pairs = key_tokens.shape[1]

        # Embed everything
        query_embed = self.token_embed(query_tokens)  # (batch, d_model)
        key_embeds = self.token_embed(key_tokens)     # (batch, num_pairs, d_model)
        value_embeds = self.token_embed(value_tokens) # (batch, num_pairs, d_model)

        # Create sequence: [query, key1, val1, key2, val2, ...]
        # But for simplicity, we'll do direct attention from query to keys

        # Compute attention: query attends to all keys
        # This is exactly what induction heads do!
        scores = mx.matmul(query_embed[:, None, :], key_embeds.transpose(0, 2, 1))
        scores = scores.squeeze(1) / (self.config.d_model ** 0.5)  # (batch, num_pairs)

        # Hard attention with low temperature
        attention = mx.softmax(scores / 0.01, axis=-1)

        # Retrieve value based on attention
        retrieved = mx.matmul(attention[:, None, :], value_embeds)  # (batch, 1, d_model)
        retrieved = retrieved.squeeze(1)  # (batch, d_model)

        # Project to vocabulary
        logits = self.output_proj(retrieved)  # (batch, max_vocab)

        return logits, attention


class TransformerLayer(nn.Module):
    """Simple transformer layer."""

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
        x = self.norm1(x + self.attention(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x


class Tokenizer:
    """Simple tokenizer that maps 4-char strings to unique IDs."""

    def __init__(self):
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: Dict[int, str] = {}
        self.next_id = 1  # 0 reserved for padding

    def encode(self, s: str) -> int:
        """Get or create token ID for a string."""
        if s not in self.str_to_id:
            self.str_to_id[s] = self.next_id
            self.id_to_str[self.next_id] = s
            self.next_id += 1
        return self.str_to_id[s]

    def decode(self, token_id: int) -> str:
        """Convert token ID back to string."""
        return self.id_to_str.get(token_id, "")

    def vocab_size(self) -> int:
        return self.next_id


def parse_context(context: str) -> List[Tuple[str, str]]:
    """Parse HashHop context into (key, value) pairs."""
    pairs = []
    pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
    for match in re.finditer(pattern, context):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def prepare_batch(
    samples: List[Tuple[str, str, str]],  # (context, query, target)
    tokenizer: Tokenizer,
    max_pairs: int,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Prepare tokenized batch."""
    batch_queries = []
    batch_keys = []
    batch_values = []
    batch_targets = []

    for context, query, target in samples:
        pairs = parse_context(context)

        # Tokenize query
        query_id = tokenizer.encode(query)
        batch_queries.append(query_id)

        # Tokenize keys and values
        key_ids = []
        value_ids = []
        for k, v in pairs[:max_pairs]:
            key_ids.append(tokenizer.encode(k))
            value_ids.append(tokenizer.encode(v))

        # Pad
        while len(key_ids) < max_pairs:
            key_ids.append(0)
            value_ids.append(0)

        batch_keys.append(key_ids)
        batch_values.append(value_ids)

        # Tokenize target
        target_id = tokenizer.encode(target)
        batch_targets.append(target_id)

    return (
        mx.array(batch_queries, dtype=mx.int32),
        mx.array(batch_keys, dtype=mx.int32),
        mx.array(batch_values, dtype=mx.int32),
        mx.array(batch_targets, dtype=mx.int32),
    )


def train_and_evaluate(
    context_size: int,
    max_steps: int = 3000,
    batch_size: int = 16,
    eval_every: int = 500,
):
    """Train and evaluate tokenized hash table."""
    print(f"\n{'='*60}")
    print(f"TOKENIZED Hash Table: {context_size:,} chars ({context_size // 10:,} pairs)")
    print(f"Each 4-char string = 1 token (like MQAR)")
    print(f"{'='*60}")

    config = TokenConfig()
    model = TokenizedHashTable(config)
    tokenizer = Tokenizer()

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

    print(f"Model parameters: {count_params(model.parameters()):,}")

    eval_gen = MultiHopEval()
    optimizer = optim.AdamW(learning_rate=1e-3)

    def loss_fn(params, query, keys, values, targets):
        model.update(params)
        logits, _ = model(query, keys, values)  # (batch, vocab)

        # Cross-entropy loss
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        batch_idx = mx.arange(logits.shape[0])
        target_log_probs = log_probs[batch_idx, targets]
        return -target_log_probs.mean()

    loss_and_grad = mx.value_and_grad(loss_fn)

    print(f"\nTraining for {max_steps} steps...")
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

        query, keys, values, targets = prepare_batch(samples, tokenizer, max_pairs)

        params = model.parameters()
        loss, grads = loss_and_grad(params, query, keys, values, targets)

        new_params = optimizer.apply_gradients(grads, params)
        model.update(new_params)
        mx.eval(model.parameters(), optimizer.state)

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}: loss={float(loss):.4f}, vocab={tokenizer.vocab_size()}, time={elapsed:.0f}s")

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
                        [(sample.prompt, q, expected)], tokenizer, max_pairs
                    )
                    logits, attn = model(query_b, keys_b, values_b)
                    pred_id = int(mx.argmax(logits[0]).item())
                    pred_str = tokenizer.decode(pred_id)

                    max_attn = float(mx.max(attn[0]).item())

                    if pred_str == expected:
                        correct += 1
                    break

            accuracy = correct / total * 100
            if accuracy > best_acc:
                best_acc = accuracy
            print(f"  Eval: {accuracy:.0f}% (best: {best_acc:.0f}%), max_attn: {max_attn:.4f}")

    # Final evaluation
    print(f"\nFinal evaluation on 100 samples...")
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
                [(sample.prompt, q, expected)], tokenizer, max_pairs
            )
            logits, _ = model(query_b, keys_b, values_b)
            pred_id = int(mx.argmax(logits[0]).item())
            pred_str = tokenizer.decode(pred_id)

            if pred_str == expected:
                correct += 1
            break

    accuracy = correct / total * 100
    print(f"\nFINAL: {context_size:,} chars = {accuracy:.0f}%")
    print(f"Vocabulary size: {tokenizer.vocab_size():,} tokens")
    return accuracy


if __name__ == "__main__":
    results = {}

    # Test at various scales
    for size in [2000, 5000, 10000, 50000, 100000]:
        if size <= 10000:
            max_steps = 3000
            batch_size = 16
        else:
            max_steps = 5000
            batch_size = 8

        acc = train_and_evaluate(size, max_steps=max_steps, batch_size=batch_size)
        results[size] = acc

    print("\n" + "="*60)
    print("TOKENIZED Hash Table RESULTS")
    print("(Each 4-char string = 1 token, like MQAR)")
    print("="*60)
    for ctx, acc in sorted(results.items()):
        print(f"  {ctx:>10,} chars ({ctx//10:>6,} pairs): {acc:.0f}%")
