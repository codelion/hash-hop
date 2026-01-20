"""Tokenized HashHop Solver - MLX GPU-accelerated version.

This uses Apple Silicon's Metal GPU via MLX for significant speedup
over the NumPy CPU implementation.

Key optimizations:
1. GPU-accelerated attention computation
2. Batch processing for efficiency
3. Lazy evaluation to minimize memory transfers
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple, Optional
import re
import time
import argparse

from hashhop import MultiHopEval


class HashTokenizer:
    """Maps hash strings to unique token IDs."""

    def __init__(self):
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: Dict[int, str] = {}
        self.next_id = 1  # 0 reserved for padding

    def encode(self, s: str) -> int:
        if s not in self.str_to_id:
            self.str_to_id[s] = self.next_id
            self.id_to_str[self.next_id] = s
            self.next_id += 1
        return self.str_to_id[s]

    def decode(self, tid: int) -> str:
        return self.id_to_str.get(tid, "")

    def vocab_size(self) -> int:
        return self.next_id


class MLXTokenizedRetriever(nn.Module):
    """GPU-accelerated token-based associative memory.

    Uses MLX for Apple Silicon GPU acceleration.
    """

    def __init__(self, d_model: int = 128, initial_vocab: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.current_vocab = initial_vocab
        # Embeddings stored as MLX array
        self.embeddings = mx.random.normal((initial_vocab, d_model)) * 0.1
        self._normalize_all()

    def _normalize_all(self):
        """L2 normalize all embeddings."""
        norms = mx.sqrt(mx.sum(self.embeddings ** 2, axis=1, keepdims=True) + 1e-8)
        self.embeddings = self.embeddings / norms
        mx.eval(self.embeddings)

    def ensure_capacity(self, max_id: int):
        """Grow embedding table if needed."""
        if max_id >= self.current_vocab:
            new_size = max(max_id + 10000, self.current_vocab * 2)
            new_embeds = mx.random.normal((new_size, self.d_model)) * 0.1
            # Copy old embeddings
            new_embeds = mx.concatenate([
                self.embeddings,
                new_embeds[self.current_vocab:]
            ], axis=0)
            self.embeddings = new_embeds
            self.current_vocab = new_size
            self._normalize_all()

    def retrieve(
        self,
        query_id: int,
        key_ids: mx.array,
        value_ids: mx.array,
        temperature: float = 0.01
    ) -> Tuple[int, mx.array]:
        """Retrieve value for query via attention over keys.

        All computation on GPU.
        """
        # Get embeddings (GPU)
        q_emb = self.embeddings[query_id]  # (d,)
        k_emb = self.embeddings[key_ids]   # (n, d)
        v_emb = self.embeddings[value_ids] # (n, d)

        # Compute attention (GPU)
        scores = mx.matmul(k_emb, q_emb) / temperature  # (n,)
        scores = scores - mx.max(scores)  # Numerical stability
        attn = mx.softmax(scores)

        # Retrieve value embedding (GPU)
        out_emb = mx.matmul(attn[None, :], v_emb).squeeze(0)  # (d,)

        # Find nearest token (GPU)
        similarities = mx.matmul(self.embeddings, out_emb)
        pred_id = int(mx.argmax(similarities).item())

        return pred_id, attn

    def train_step(
        self,
        query_id: int,
        key_ids: mx.array,
        value_ids: mx.array,
        target_id: int,
        lr: float = 0.1
    ) -> float:
        """Update embeddings to improve retrieval (GPU-accelerated)."""
        # Get embeddings
        q_emb = self.embeddings[query_id]
        k_emb = self.embeddings[key_ids]
        v_emb = self.embeddings[value_ids]
        t_emb = self.embeddings[target_id]

        # Compute attention with softer temperature for training
        scores = mx.matmul(k_emb, q_emb) / 0.1
        scores = scores - mx.max(scores)
        attn = mx.softmax(scores)

        # Retrieved value embedding
        out_emb = mx.matmul(attn[None, :], v_emb).squeeze(0)

        # Loss: push out_emb closer to target embedding
        error = out_emb - t_emb
        loss = float(mx.sum(error ** 2).item())

        # Update value embeddings that were attended to
        # Create update mask based on attention weights
        attn_mask = (attn > 0.01).astype(mx.float32)

        # Compute gradients for value embeddings
        # For each attended value, move it towards target
        for i in range(len(value_ids)):
            vid = int(value_ids[i].item())
            a = float(attn[i].item())
            if a > 0.01 and vid > 0:
                update = lr * a * error
                self.embeddings = self.embeddings.at[vid].add(-update)

        # Update target embedding
        self.embeddings = self.embeddings.at[target_id].add(lr * 0.5 * error)

        # Re-normalize updated embeddings
        self._normalize_all()

        return loss


def parse_hashhop_context(context: str, hash_length: int = 16) -> List[Tuple[str, str]]:
    """Parse HashHop context into (key, value) pairs."""
    pairs = []
    pattern = rf"([a-zA-Z]{{{hash_length}}})\s*=\s*'?([a-zA-Z]{{{hash_length}}})'?"
    for match in re.finditer(pattern, context):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def train_and_evaluate(
    context_tokens: int,
    hash_length: int = 16,
    hops: int = 2,
    num_queries: int = 1,
    max_steps: int = 1000,
    eval_samples: int = 100,
    d_model: int = 128,
    verbose: bool = True
) -> float:
    """Train and evaluate MLX-accelerated tokenized HashHop solver."""
    n_chars = context_tokens * 3

    if verbose:
        print(f"\n{'='*70}")
        print(f"TOKENIZED HASHHOP (MLX GPU): {context_tokens:,} tokens ({n_chars:,} chars)")
        print(f"Hash length: {hash_length}, Hops: {hops}, d_model: {d_model}")
        print(f"{'='*70}")

    # Estimate initial vocab size based on context
    initial_vocab = min(context_tokens * 10, 10_000_000)
    model = MLXTokenizedRetriever(d_model=d_model, initial_vocab=initial_vocab)
    tokenizer = HashTokenizer()
    eval_gen = MultiHopEval()

    if verbose:
        print(f"Training for {max_steps} steps...")
    start = time.time()
    best_acc = 0

    for step in range(1, max_steps + 1):
        # Generate training sample
        sample = eval_gen.make_one(
            n_chars_problem=n_chars,
            num_queries=num_queries,
            hops=hops,
            hash_pair_str_length=hash_length,
            chain_of_thought=False,
        )
        pairs = parse_hashhop_context(sample.prompt, hash_length)

        # Train on each query
        for query, target in sample.targets.items():
            # Build chain for this query
            chain = [query]
            lookup = {k: v for k, v in pairs}
            current = query
            for _ in range(hops):
                if current in lookup:
                    current = lookup[current]
                    chain.append(current)

            # Ensure vocab capacity
            max_id = max(tokenizer.encode(s) for s in chain)
            for k, v in pairs:
                max_id = max(max_id, tokenizer.encode(k), tokenizer.encode(v))
            model.ensure_capacity(max_id + 1)

            # Train on each hop in the chain
            for i in range(len(chain) - 1):
                q_id = tokenizer.encode(chain[i])
                t_id = tokenizer.encode(chain[i + 1])

                key_ids = mx.array([tokenizer.encode(k) for k, v in pairs])
                val_ids = mx.array([tokenizer.encode(v) for k, v in pairs])

                model.train_step(q_id, key_ids, val_ids, t_id, lr=0.05)

        if verbose and step % 100 == 0:
            elapsed = time.time() - start
            print(f"Step {step}: vocab={tokenizer.vocab_size():,}, time={elapsed:.0f}s")

        # Periodic evaluation
        if step % 200 == 0 or step == max_steps:
            correct = 0
            total = min(50, eval_samples)

            for _ in range(total):
                sample = eval_gen.make_one(
                    n_chars_problem=n_chars,
                    num_queries=1,
                    hops=hops,
                    hash_pair_str_length=hash_length,
                    chain_of_thought=False,
                )
                pairs = parse_hashhop_context(sample.prompt, hash_length)

                for query, expected in sample.targets.items():
                    # Ensure vocab
                    for k, v in pairs:
                        model.ensure_capacity(max(tokenizer.encode(k), tokenizer.encode(v)) + 1)

                    current_id = tokenizer.encode(query)
                    key_ids = mx.array([tokenizer.encode(k) for k, v in pairs])
                    val_ids = mx.array([tokenizer.encode(v) for k, v in pairs])

                    # Multi-hop retrieval
                    for _ in range(hops):
                        pred_id, attn = model.retrieve(current_id, key_ids, val_ids)
                        current_id = pred_id

                    pred_str = tokenizer.decode(pred_id)
                    if pred_str == expected:
                        correct += 1
                    break

            acc = correct / total * 100
            if acc > best_acc:
                best_acc = acc
            if verbose:
                print(f"  Eval: {acc:.0f}% (best: {best_acc:.0f}%)")

    # Final evaluation
    if verbose:
        print(f"\nFinal evaluation on {eval_samples} samples...")

    correct = 0
    for _ in range(eval_samples):
        sample = eval_gen.make_one(
            n_chars_problem=n_chars,
            num_queries=1,
            hops=hops,
            hash_pair_str_length=hash_length,
            chain_of_thought=False,
        )
        pairs = parse_hashhop_context(sample.prompt, hash_length)

        for query, expected in sample.targets.items():
            for k, v in pairs:
                model.ensure_capacity(max(tokenizer.encode(k), tokenizer.encode(v)) + 1)

            current_id = tokenizer.encode(query)
            key_ids = mx.array([tokenizer.encode(k) for k, v in pairs])
            val_ids = mx.array([tokenizer.encode(v) for k, v in pairs])

            for _ in range(hops):
                pred_id, _ = model.retrieve(current_id, key_ids, val_ids)
                current_id = pred_id

            pred_str = tokenizer.decode(pred_id)
            if pred_str == expected:
                correct += 1
            break

    accuracy = correct / eval_samples * 100
    elapsed = time.time() - start

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: {context_tokens:,} tokens = {accuracy:.0f}% accuracy")
        print(f"Vocab size: {tokenizer.vocab_size():,} tokens")
        print(f"Total time: {elapsed:.0f}s")
        print(f"{'='*70}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Tokenized HashHop Solver (MLX GPU)")
    parser.add_argument("--tokens", type=int, default=1000,
                        help="Context size in tokens (default: 1000)")
    parser.add_argument("--hash-length", type=int, default=16,
                        help="Hash string length (default: 16)")
    parser.add_argument("--hops", type=int, default=2,
                        help="Number of hops (default: 2)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps (default: 1000)")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Evaluation samples (default: 100)")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Embedding dimension (default: 128)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark across scales")
    args = parser.parse_args()

    if args.benchmark:
        results = {}
        scales = [
            (1_000, 500),
            (10_000, 500),
            (100_000, 500),
            (1_000_000, 500),
            (10_000_000, 500),
            (100_000_000, 500),
        ]

        print("\n" + "="*70)
        print("TOKENIZED HASHHOP BENCHMARK (MLX GPU)")
        print(f"Hash length: {args.hash_length}, Hops: {args.hops}")
        print("="*70)

        for tokens, steps in scales:
            acc = train_and_evaluate(
                context_tokens=tokens,
                hash_length=args.hash_length,
                hops=args.hops,
                max_steps=steps,
                eval_samples=50,
                d_model=args.d_model,
                verbose=True
            )
            results[tokens] = acc

        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Context':>15} | {'Accuracy':>10} | {'Gemini Flash':>12}")
        print("-" * 45)

        gemini_results = {
            1_000: 100,
            10_000: 96,
            100_000: 77,
            200_000: 37,
            500_000: 9,
            1_000_000: 4,
        }

        for tokens, acc in sorted(results.items()):
            gemini = gemini_results.get(tokens, "-")
            gemini_str = f"{gemini}%" if isinstance(gemini, int) else gemini
            print(f"{tokens:>12,} T | {acc:>9.0f}% | {gemini_str:>12}")

    else:
        train_and_evaluate(
            context_tokens=args.tokens,
            hash_length=args.hash_length,
            hops=args.hops,
            max_steps=args.steps,
            eval_samples=args.eval_samples,
            d_model=args.d_model,
            verbose=True
        )


if __name__ == "__main__":
    main()
