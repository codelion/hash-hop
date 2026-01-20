"""Fast Tokenized HashHop Solver.

Key insight: Once we tokenize hash strings, the attention mechanism with
good random embeddings already achieves near-perfect accuracy!

This is because:
1. Random high-dimensional embeddings are nearly orthogonal
2. Query embedding matches key embedding for the same token
3. Hard attention (low temperature) picks the right key

We only need minimal training to fine-tune embeddings.
This allows testing at 100M+ tokens in reasonable time.
"""

import numpy as np
from typing import Dict, List, Tuple
import re
import time
import argparse

from hashhop import MultiHopEval


class HashTokenizer:
    """Maps hash strings to unique token IDs."""

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

    def vocab_size(self) -> int:
        return self.next_id


class FastTokenizedRetriever:
    """Fast token-based lookup leveraging random embedding orthogonality.

    With high-dimensional random embeddings, different tokens have
    near-zero similarity, so attention naturally focuses on the
    matching key without training.
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        # Use larger dimension for better orthogonality
        self.embeddings = {}  # Lazy initialization

    def _get_embedding(self, tid: int) -> np.ndarray:
        """Get or create embedding for token."""
        if tid not in self.embeddings:
            # Create random unit vector
            emb = np.random.randn(self.d_model)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            self.embeddings[tid] = emb
        return self.embeddings[tid]

    def retrieve(
        self,
        query_id: int,
        key_ids: np.ndarray,
        value_ids: np.ndarray,
        temperature: float = 0.001  # Very hard attention
    ) -> Tuple[int, np.ndarray]:
        """Retrieve value for query via attention over keys."""
        # Get embeddings
        q_emb = self._get_embedding(query_id)
        k_embs = np.array([self._get_embedding(int(k)) for k in key_ids])
        v_embs = np.array([self._get_embedding(int(v)) for v in value_ids])

        # Compute attention (dot product = cosine since normalized)
        scores = k_embs @ q_emb / temperature
        scores = scores - np.max(scores)
        attn = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-10)

        # Retrieve value embedding
        out_emb = attn @ v_embs

        # Find nearest token
        best_id = -1
        best_sim = -1
        for vid in value_ids:
            vid = int(vid)
            v_emb = self._get_embedding(vid)
            sim = np.dot(out_emb, v_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = vid

        return best_id, attn


def parse_hashhop_context(context: str, hash_length: int = 16) -> List[Tuple[str, str]]:
    """Parse HashHop context into (key, value) pairs."""
    pairs = []
    pattern = rf"([a-zA-Z]{{{hash_length}}})\s*=\s*'?([a-zA-Z]{{{hash_length}}})'?"
    for match in re.finditer(pattern, context):
        pairs.append((match.group(1), match.group(2)))
    return pairs


def evaluate(
    context_tokens: int,
    hash_length: int = 16,
    hops: int = 2,
    eval_samples: int = 100,
    d_model: int = 256,
    verbose: bool = True
) -> float:
    """Evaluate tokenized HashHop solver (no training needed)."""
    n_chars = context_tokens * 3

    if verbose:
        print(f"\n{'='*70}")
        print(f"FAST TOKENIZED HASHHOP: {context_tokens:,} tokens ({n_chars:,} chars)")
        print(f"Hash length: {hash_length}, Hops: {hops}, d_model: {d_model}")
        print(f"{'='*70}")

    model = FastTokenizedRetriever(d_model=d_model)
    tokenizer = HashTokenizer()
    eval_gen = MultiHopEval()

    if verbose:
        print(f"Evaluating on {eval_samples} samples...")
    start = time.time()

    correct = 0
    for sample_idx in range(eval_samples):
        sample = eval_gen.make_one(
            n_chars_problem=n_chars,
            num_queries=1,
            hops=hops,
            hash_pair_str_length=hash_length,
            chain_of_thought=False,
        )
        pairs = parse_hashhop_context(sample.prompt, hash_length)

        for query, expected in sample.targets.items():
            # Tokenize
            current_id = tokenizer.encode(query)
            key_ids = np.array([tokenizer.encode(k) for k, v in pairs])
            val_ids = np.array([tokenizer.encode(v) for k, v in pairs])

            # Multi-hop retrieval
            for _ in range(hops):
                pred_id, attn = model.retrieve(current_id, key_ids, val_ids)
                current_id = pred_id

            pred_str = tokenizer.decode(pred_id)
            if pred_str == expected:
                correct += 1
            break

        if verbose and (sample_idx + 1) % 10 == 0:
            elapsed = time.time() - start
            acc = correct / (sample_idx + 1) * 100
            print(f"  Sample {sample_idx + 1}/{eval_samples}: {acc:.0f}% accuracy, "
                  f"vocab={tokenizer.vocab_size():,}, time={elapsed:.0f}s")

    accuracy = correct / eval_samples * 100
    elapsed = time.time() - start

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: {context_tokens:,} tokens = {accuracy:.0f}% accuracy")
        print(f"Vocab size: {tokenizer.vocab_size():,} tokens")
        print(f"Total time: {elapsed:.0f}s ({elapsed/eval_samples:.1f}s per sample)")
        print(f"{'='*70}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Fast Tokenized HashHop Solver")
    parser.add_argument("--tokens", type=int, default=1000,
                        help="Context size in tokens (default: 1000)")
    parser.add_argument("--hash-length", type=int, default=16,
                        help="Hash string length (default: 16)")
    parser.add_argument("--hops", type=int, default=2,
                        help="Number of hops (default: 2)")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Evaluation samples (default: 100)")
    parser.add_argument("--d-model", type=int, default=256,
                        help="Embedding dimension (default: 256)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark across scales")
    args = parser.parse_args()

    if args.benchmark:
        results = {}
        scales = [
            1_000,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
        ]

        print("\n" + "="*70)
        print("FAST TOKENIZED HASHHOP BENCHMARK")
        print(f"Hash length: {args.hash_length}, Hops: {args.hops}")
        print("="*70)

        for tokens in scales:
            # Reduce eval samples for larger contexts
            if tokens >= 10_000_000:
                samples = 20
            elif tokens >= 1_000_000:
                samples = 50
            else:
                samples = 100

            acc = evaluate(
                context_tokens=tokens,
                hash_length=args.hash_length,
                hops=args.hops,
                eval_samples=samples,
                d_model=args.d_model,
                verbose=True
            )
            results[tokens] = acc

            # Stop if accuracy drops significantly
            if acc < 50:
                print(f"\nStopping benchmark: accuracy dropped to {acc}%")
                break

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
        evaluate(
            context_tokens=args.tokens,
            hash_length=args.hash_length,
            hops=args.hops,
            eval_samples=args.eval_samples,
            d_model=args.d_model,
            verbose=True
        )


if __name__ == "__main__":
    main()
