"""Simple Tokenized Hash Table for HashHop.

Minimal implementation: Each 4-char string = 1 token.
Direct embedding comparison for lookup (no transformer layers needed).

This is exactly MQAR - which attention solves perfectly!
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
class Config:
    d_model: int = 64  # Small embedding dim
    max_vocab: int = 2000  # Max unique tokens


class SimpleTokenLookup(nn.Module):
    """Minimal token-based lookup - just embeddings + attention."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.max_vocab, config.d_model)

    def __call__(self, query, keys, values):
        """
        query: (batch,) token IDs
        keys: (batch, num_pairs) token IDs
        values: (batch, num_pairs) token IDs
        """
        # Embed
        q_emb = self.token_embed(query)      # (batch, d)
        k_emb = self.token_embed(keys)       # (batch, n, d)
        v_emb = self.token_embed(values)     # (batch, n, d)

        # Attention: query dot-product with keys
        scores = mx.sum(q_emb[:, None, :] * k_emb, axis=-1)  # (batch, n)
        scores = scores / (self.config.d_model ** 0.5)

        # Hard attention
        attn = mx.softmax(scores / 0.01, axis=-1)

        # Retrieve value embedding
        out = mx.sum(attn[:, :, None] * v_emb, axis=1)  # (batch, d)

        # Predict token by finding nearest embedding
        # Compute similarity with all token embeddings
        all_embeds = self.token_embed.weight  # (vocab, d)
        logits = mx.matmul(out, all_embeds.T)  # (batch, vocab)

        return logits, attn


class Tokenizer:
    def __init__(self):
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: Dict[int, str] = {}
        self.next_id = 1

    def encode(self, s: str) -> int:
        if s not in self.str_to_id:
            self.str_to_id[s] = self.next_id
            self.id_to_str[self.next_id] = s
            self.next_id += 1
        return self.str_to_id[s]

    def decode(self, tid: int) -> str:
        return self.id_to_str.get(tid, "")


def parse_context(context: str) -> List[Tuple[str, str]]:
    pairs = []
    pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
    for match in re.finditer(pattern, context):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def prepare_batch(samples, tokenizer, max_pairs):
    queries, keys, values, targets = [], [], [], []

    for context, query, target in samples:
        pairs = parse_context(context)

        queries.append(tokenizer.encode(query))
        targets.append(tokenizer.encode(target))

        ks, vs = [], []
        for k, v in pairs[:max_pairs]:
            ks.append(tokenizer.encode(k))
            vs.append(tokenizer.encode(v))
        while len(ks) < max_pairs:
            ks.append(0)
            vs.append(0)
        keys.append(ks)
        values.append(vs)

    return (
        mx.array(queries, dtype=mx.int32),
        mx.array(keys, dtype=mx.int32),
        mx.array(values, dtype=mx.int32),
        mx.array(targets, dtype=mx.int32),
    )


def train_and_evaluate(context_size, max_steps=2000, batch_size=32):
    print(f"\n{'='*60}")
    print(f"TOKENIZED: {context_size:,} chars ({context_size // 10:,} pairs)")
    print(f"{'='*60}")

    config = Config()
    model = SimpleTokenLookup(config)
    tokenizer = Tokenizer()
    max_pairs = context_size // 10 + 10

    eval_gen = MultiHopEval()
    optimizer = optim.Adam(learning_rate=1e-2)

    def loss_fn(params, q, k, v, t):
        model.update(params)
        logits, _ = model(q, k, v)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        idx = mx.arange(logits.shape[0])
        return -log_probs[idx, t].mean()

    loss_and_grad = mx.value_and_grad(loss_fn)

    print(f"Training...")
    start = time.time()
    best = 0

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

        qb, kb, vb, tb = prepare_batch(samples, tokenizer, max_pairs)

        params = model.parameters()
        loss, grads = loss_and_grad(params, qb, kb, vb, tb)
        new_params = optimizer.apply_gradients(grads, params)
        model.update(new_params)
        mx.eval(model.parameters(), optimizer.state)

        if step % 100 == 0:
            print(f"Step {step}: loss={float(loss):.4f}, vocab={tokenizer.next_id}")

        if step % 500 == 0:
            correct = 0
            for _ in range(50):
                sample = eval_gen.make_one(
                    n_chars_problem=context_size,
                    num_queries=1,
                    hops=1,
                    hash_pair_str_length=4,
                    chain_of_thought=False,
                )
                for q, exp in sample.targets.items():
                    qb, kb, vb, _ = prepare_batch([(sample.prompt, q, exp)], tokenizer, max_pairs)
                    logits, attn = model(qb, kb, vb)
                    pred = tokenizer.decode(int(mx.argmax(logits[0]).item()))
                    if pred == exp:
                        correct += 1
                    break
            acc = correct / 50 * 100
            if acc > best:
                best = acc
            print(f"  Eval: {acc:.0f}% (best: {best:.0f}%), max_attn: {float(mx.max(attn).item()):.4f}")

    # Final
    correct = 0
    for _ in range(100):
        sample = eval_gen.make_one(
            n_chars_problem=context_size,
            num_queries=1,
            hops=1,
            hash_pair_str_length=4,
            chain_of_thought=False,
        )
        for q, exp in sample.targets.items():
            qb, kb, vb, _ = prepare_batch([(sample.prompt, q, exp)], tokenizer, max_pairs)
            logits, _ = model(qb, kb, vb)
            pred = tokenizer.decode(int(mx.argmax(logits[0]).item()))
            if pred == exp:
                correct += 1
            break

    print(f"\nFINAL: {context_size:,} chars = {correct}%")
    return correct


if __name__ == "__main__":
    results = {}
    for size in [2000, 5000, 10000, 50000]:
        acc = train_and_evaluate(size, max_steps=2000, batch_size=32)
        results[size] = acc

    print("\n" + "="*60)
    print("TOKENIZED RESULTS (each 4-char string = 1 token)")
    print("="*60)
    for ctx, acc in sorted(results.items()):
        print(f"  {ctx:>10,} chars: {acc}%")
