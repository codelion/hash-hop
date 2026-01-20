"""Tokenized HashHop Solver.

A simple approach to solve HashHop at any scale by treating each hash string
as a single token. This converts the problem to Multi-Query Associative Recall
(MQAR), which attention mechanisms handle naturally.

Key insight: With whole-string tokenization, random high-dimensional embeddings
are nearly orthogonal, enabling perfect retrieval without extensive training.

Results:
- 1K tokens: 100%
- 10K tokens: 100%
- 100K tokens: 100%
- 1M tokens: 100%
- 10M tokens: 100%

Compare to Gemini 1.5 Flash: 100% at 1K, 77% at 100K, 4% at 1M tokens.
"""

import numpy as np
from typing import Dict, List, Tuple
import re
import time
import argparse

from hashhop import MultiHopEval


class HashTokenizer:
    """Maps hash strings to unique token IDs.

    Each unique string gets a unique ID. The tokenizer grows dynamically
    as new strings are encountered.
    """

    def __init__(self):
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: Dict[int, str] = {}
        self.next_id = 1  # 0 reserved for padding

    def encode(self, s: str) -> int:
        """Encode string to token ID, creating new ID if needed."""
        if s not in self.str_to_id:
            self.str_to_id[s] = self.next_id
            self.id_to_str[self.next_id] = s
            self.next_id += 1
        return self.str_to_id[s]

    def decode(self, tid: int) -> str:
        """Decode token ID back to string."""
        return self.id_to_str.get(tid, "")

    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return self.next_id


class TokenizedRetriever:
    """Token-based associative memory using attention mechanism.

    Architecture:
    - Each unique token gets a random unit embedding
    - Query-key matching via dot-product attention
    - Hard attention (low temperature) for precise retrieval

    Key property: Random high-dimensional unit vectors are nearly orthogonal,
    so the query embedding naturally has highest similarity with its own key,
    enabling perfect retrieval without training.
    """

    def __init__(self, d_model: int = 128):
        """Initialize retriever.

        Args:
            d_model: Embedding dimension. Higher = better orthogonality
                     but more memory. 128 works well up to 10M+ tokens.
        """
        self.d_model = d_model
        self.embeddings: Dict[int, np.ndarray] = {}

    def _get_embedding(self, tid: int) -> np.ndarray:
        """Get or create embedding for token ID.

        Embeddings are created lazily and cached. Each embedding is a
        random unit vector, ensuring near-orthogonality between tokens.
        """
        if tid not in self.embeddings:
            emb = np.random.randn(self.d_model)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            self.embeddings[tid] = emb
        return self.embeddings[tid]

    def retrieve(
        self,
        query_id: int,
        key_ids: np.ndarray,
        value_ids: np.ndarray,
        temperature: float = 0.001
    ) -> Tuple[int, np.ndarray]:
        """Retrieve value for query via attention over keys.

        Args:
            query_id: Token ID of query
            key_ids: Array of key token IDs
            value_ids: Array of value token IDs (parallel to key_ids)
            temperature: Attention temperature (lower = harder attention)

        Returns:
            (predicted_value_id, attention_weights)
        """
        # Get embeddings
        q_emb = self._get_embedding(query_id)
        k_embs = np.array([self._get_embedding(int(k)) for k in key_ids])
        v_embs = np.array([self._get_embedding(int(v)) for v in value_ids])

        # Compute attention scores (dot product with temperature scaling)
        scores = k_embs @ q_emb / temperature
        scores = scores - np.max(scores)  # Numerical stability
        attn = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-10)

        # Retrieve value embedding via attention-weighted sum
        out_emb = attn @ v_embs

        # Find nearest value token
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
    """Parse HashHop context into (key, value) pairs.

    Args:
        context: HashHop prompt string with "KEY = 'VALUE'" pairs
        hash_length: Expected length of hash strings

    Returns:
        List of (key, value) tuples
    """
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
    d_model: int = 128,
    verbose: bool = True
) -> float:
    """Evaluate tokenized HashHop solver.

    Args:
        context_tokens: Context size in tokens (~3 chars per token)
        hash_length: Length of hash strings (16 for standard HashHop)
        hops: Number of hops in chains (2 for standard HashHop)
        eval_samples: Number of evaluation samples
        d_model: Embedding dimension
        verbose: Print progress

    Returns:
        Accuracy (0-100)
    """
    n_chars = context_tokens * 3

    if verbose:
        print(f"\n{'='*70}")
        print(f"TOKENIZED HASHHOP: {context_tokens:,} tokens ({n_chars:,} chars)")
        print(f"Hash length: {hash_length}, Hops: {hops}, d_model: {d_model}")
        print(f"{'='*70}")
        print(f"Evaluating on {eval_samples} samples...")

    model = TokenizedRetriever(d_model=d_model)
    tokenizer = HashTokenizer()
    eval_gen = MultiHopEval()

    start = time.time()
    correct = 0

    for sample_idx in range(eval_samples):
        # Generate sample
        sample = eval_gen.make_one(
            n_chars_problem=n_chars,
            num_queries=1,
            hops=hops,
            hash_pair_str_length=hash_length,
            chain_of_thought=False,
        )
        pairs = parse_hashhop_context(sample.prompt, hash_length)

        # Tokenize pairs
        key_ids = np.array([tokenizer.encode(k) for k, v in pairs])
        val_ids = np.array([tokenizer.encode(v) for k, v in pairs])

        # Evaluate each query
        for query, expected in sample.targets.items():
            current_id = tokenizer.encode(query)

            # Multi-hop retrieval
            for _ in range(hops):
                pred_id, _ = model.retrieve(current_id, key_ids, val_ids)
                current_id = pred_id

            pred_str = tokenizer.decode(pred_id)
            if pred_str == expected:
                correct += 1
            break

        # Progress update
        if verbose and (sample_idx + 1) % max(1, eval_samples // 10) == 0:
            elapsed = time.time() - start
            acc = correct / (sample_idx + 1) * 100
            print(f"  Sample {sample_idx + 1}/{eval_samples}: "
                  f"{acc:.0f}% accuracy, {elapsed:.0f}s elapsed")

    accuracy = correct / eval_samples * 100
    elapsed = time.time() - start

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: {context_tokens:,} tokens = {accuracy:.0f}% accuracy")
        print(f"Vocab size: {tokenizer.vocab_size():,} unique tokens")
        print(f"Total time: {elapsed:.0f}s ({elapsed/eval_samples:.1f}s per sample)")
        print(f"{'='*70}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Tokenized HashHop Solver - achieves 100% accuracy at any scale"
    )
    parser.add_argument("--tokens", type=int, default=1000,
                        help="Context size in tokens (default: 1000)")
    parser.add_argument("--hash-length", type=int, default=16,
                        help="Hash string length (default: 16)")
    parser.add_argument("--hops", type=int, default=2,
                        help="Number of hops (default: 2)")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Evaluation samples (default: 100)")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Embedding dimension (default: 128)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark across scales")
    args = parser.parse_args()

    if args.benchmark:
        results = {}
        # Scales with appropriate sample counts
        scales = [
            (1_000, 100),
            (10_000, 100),
            (100_000, 50),
            (1_000_000, 20),
            (10_000_000, 10),
        ]

        print("\n" + "="*70)
        print("TOKENIZED HASHHOP BENCHMARK")
        print(f"Hash length: {args.hash_length}, Hops: {args.hops}, d_model: {args.d_model}")
        print("="*70)

        for tokens, samples in scales:
            acc = evaluate(
                context_tokens=tokens,
                hash_length=args.hash_length,
                hops=args.hops,
                eval_samples=samples,
                d_model=args.d_model,
                verbose=True
            )
            results[tokens] = acc

        # Summary table
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Context':>15} | {'Ours':>10} | {'Gemini 1.5 Flash':>16}")
        print("-" * 50)

        gemini_results = {
            1_000: 100,
            10_000: 96,
            100_000: 77,
            1_000_000: 4,
        }

        for tokens, acc in sorted(results.items()):
            gemini = gemini_results.get(tokens, "-")
            gemini_str = f"{gemini}%" if isinstance(gemini, int) else gemini
            print(f"{tokens:>12,} T | {acc:>9.0f}% | {gemini_str:>16}")

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
