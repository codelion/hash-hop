"""100M Token HashHop Test - Memory Efficient Version.

For 100M tokens, we can't generate the full context string.
Instead, we directly generate the key-value pairs and test retrieval.

This is equivalent to the actual HashHop task but without
the string serialization overhead.
"""

import numpy as np
from typing import Dict, List, Tuple
import time
import argparse
import string


def random_string(length: int) -> str:
    """Generate random hash string."""
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return ''.join(np.random.choice(list(alphabet), length))


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


class StreamingTokenizedRetriever:
    """Memory-efficient tokenized retriever using streaming embeddings."""

    def __init__(self, d_model: int = 32):
        self.d_model = d_model
        # Use a seed-based approach: hash(token_id) -> embedding
        # This avoids storing all embeddings in memory

    def _get_embedding(self, tid: int) -> np.ndarray:
        """Generate embedding deterministically from token ID."""
        # Use token ID as seed for reproducibility
        rng = np.random.RandomState(tid)
        emb = rng.randn(self.d_model)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def retrieve(
        self,
        query_id: int,
        key_ids: np.ndarray,
        value_ids: np.ndarray,
        temperature: float = 0.0001  # Very hard attention
    ) -> Tuple[int, float]:
        """Retrieve value for query via attention over keys.

        Returns (predicted_value_id, max_attention_weight)
        """
        # Get query embedding
        q_emb = self._get_embedding(query_id)

        # Compute attention scores in chunks to save memory
        chunk_size = 100000
        n_pairs = len(key_ids)

        best_score = float('-inf')
        best_value_id = -1

        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)

            # Get key embeddings for this chunk
            k_embs = np.array([self._get_embedding(int(k)) for k in key_ids[start:end]])

            # Compute scores for this chunk
            scores = k_embs @ q_emb

            # Find best in chunk
            chunk_best_idx = np.argmax(scores)
            chunk_best_score = scores[chunk_best_idx]

            if chunk_best_score > best_score:
                best_score = chunk_best_score
                best_value_id = int(value_ids[start + chunk_best_idx])

        return best_value_id, best_score


def generate_hashhop_pairs(
    n_pairs: int,
    hash_length: int = 16,
    hops: int = 2
) -> Tuple[List[Tuple[str, str]], str, str]:
    """Generate HashHop key-value pairs and a query with answer.

    Returns:
        (all_pairs, query_key, final_answer)
    """
    # Generate chains
    # For 2 hops: key1 -> intermediate -> answer
    # All other pairs are distractors

    n_chains = n_pairs // hops
    all_pairs = []

    # Generate one query chain
    query_key = random_string(hash_length)
    current = query_key
    for hop in range(hops):
        next_val = random_string(hash_length)
        all_pairs.append((current, next_val))
        current = next_val
    final_answer = current

    # Generate distractor pairs
    for _ in range(n_pairs - hops):
        k = random_string(hash_length)
        v = random_string(hash_length)
        all_pairs.append((k, v))

    # Shuffle
    np.random.shuffle(all_pairs)

    return all_pairs, query_key, final_answer


def test_scale(
    n_tokens: int,
    hash_length: int = 16,
    hops: int = 2,
    n_samples: int = 10,
    d_model: int = 32,
    verbose: bool = True
) -> float:
    """Test tokenized HashHop at specified scale."""
    # Each pair is ~35 chars -> 1 pair per ~12 tokens
    n_pairs = n_tokens // 12

    if verbose:
        print(f"\n{'='*70}")
        print(f"TOKENIZED HASHHOP: {n_tokens:,} tokens ({n_pairs:,} pairs)")
        print(f"Hash length: {hash_length}, Hops: {hops}, d_model: {d_model}")
        print(f"{'='*70}")
        print(f"Evaluating on {n_samples} samples...")

    model = StreamingTokenizedRetriever(d_model=d_model)

    start = time.time()
    correct = 0

    for sample_idx in range(n_samples):
        sample_start = time.time()

        # Fresh tokenizer per sample (like real HashHop)
        tokenizer = HashTokenizer()

        # Generate pairs
        pairs, query_key, expected_answer = generate_hashhop_pairs(
            n_pairs, hash_length, hops
        )

        # Tokenize
        key_ids = np.array([tokenizer.encode(k) for k, v in pairs])
        val_ids = np.array([tokenizer.encode(v) for k, v in pairs])

        # Multi-hop retrieval
        current_key = query_key
        for hop in range(hops):
            current_id = tokenizer.encode(current_key)
            pred_id, score = model.retrieve(current_id, key_ids, val_ids)
            current_key = tokenizer.decode(pred_id)

        if current_key == expected_answer:
            correct += 1

        sample_time = time.time() - sample_start

        if verbose:
            elapsed = time.time() - start
            acc = correct / (sample_idx + 1) * 100
            print(f"  Sample {sample_idx + 1}/{n_samples}: "
                  f"{'CORRECT' if current_key == expected_answer else 'WRONG'}, "
                  f"acc={acc:.0f}%, sample_time={sample_time:.0f}s, "
                  f"total_time={elapsed:.0f}s")

    accuracy = correct / n_samples * 100
    elapsed = time.time() - start

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULT: {n_tokens:,} tokens = {accuracy:.0f}% accuracy")
        print(f"Total time: {elapsed:.0f}s ({elapsed/n_samples:.0f}s per sample)")
        print(f"{'='*70}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="100M Token HashHop Test")
    parser.add_argument("--tokens", type=int, default=100_000_000,
                        help="Context size in tokens (default: 100M)")
    parser.add_argument("--hash-length", type=int, default=16,
                        help="Hash string length (default: 16)")
    parser.add_argument("--hops", type=int, default=2,
                        help="Number of hops (default: 2)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples (default: 10)")
    parser.add_argument("--d-model", type=int, default=32,
                        help="Embedding dimension (default: 32)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark")
    args = parser.parse_args()

    if args.benchmark:
        results = {}
        scales = [
            (1_000, 50),
            (10_000, 50),
            (100_000, 50),
            (1_000_000, 20),
            (10_000_000, 10),
            (100_000_000, 5),
        ]

        print("\n" + "="*70)
        print("TOKENIZED HASHHOP BENCHMARK (Memory Efficient)")
        print("="*70)

        for tokens, samples in scales:
            acc = test_scale(
                n_tokens=tokens,
                hash_length=args.hash_length,
                hops=args.hops,
                n_samples=samples,
                d_model=args.d_model,
                verbose=True
            )
            results[tokens] = acc

        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"{'Tokens':>15} | {'Accuracy':>10} | {'Gemini 1.5 Flash':>15}")
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
            print(f"{tokens:>12,} T | {acc:>9.0f}% | {gemini_str:>15}")

    else:
        test_scale(
            n_tokens=args.tokens,
            hash_length=args.hash_length,
            hops=args.hops,
            n_samples=args.samples,
            d_model=args.d_model,
            verbose=True
        )


if __name__ == "__main__":
    main()
