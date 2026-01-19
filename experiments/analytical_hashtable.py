"""Analytical Neural Hash Table - No Training Required.

Key insight: If we use deterministic character embeddings (one-hot or learned but fixed),
we can compute exact hash table lookup without any training.

This tests the upper bound of what's possible with a neural-style approach.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import List, Tuple
import re
import time

from hashhop import MultiHopEval


@dataclass
class AnalyticalConfig:
    """Configuration for analytical hash table."""
    vocab_size: int = 128
    key_length: int = 4


class AnalyticalHashTable:
    """Hash table using analytical (non-learned) embeddings.

    Uses one-hot character encoding + exact matching.
    This should achieve 100% accuracy at any scale.
    """

    def __init__(self, config: AnalyticalConfig = None):
        self.config = config or AnalyticalConfig()

    def encode_key(self, key: str) -> mx.array:
        """Encode a 4-char key to a one-hot vector."""
        # Simple: just use character codes as a unique identifier
        # For exact matching, we can use the raw character codes
        codes = [ord(c) for c in key[:self.config.key_length]]
        while len(codes) < self.config.key_length:
            codes.append(0)
        return mx.array(codes, dtype=mx.int32)

    def lookup(
        self,
        query: str,
        keys: List[str],
        values: List[str],
    ) -> Tuple[str, float]:
        """Look up query in the hash table.

        Uses exact character matching - no soft attention.
        """
        query_codes = self.encode_key(query)

        # Encode all keys
        key_codes = mx.array([[ord(c) for c in k[:4]] for k in keys], dtype=mx.int32)

        # Exact match: compute if all 4 characters match
        # matches[i] = 1 if key_codes[i] == query_codes, else 0
        matches = mx.all(key_codes == query_codes, axis=1)  # (num_keys,)

        # Find the matching index
        match_idx = mx.argmax(matches.astype(mx.float32))
        match_found = matches[match_idx].item()

        if match_found:
            return values[int(match_idx.item())], 1.0
        else:
            return "", 0.0


def parse_context(context: str) -> Tuple[List[str], List[str]]:
    """Parse HashHop context into keys and values."""
    keys = []
    values = []
    pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
    for match in re.finditer(pattern, context):
        keys.append(match.group(1))
        values.append(match.group(2))
    return keys, values


def test_scale(context_size: int, num_samples: int = 100) -> float:
    """Test analytical hash table at a given scale."""
    print(f"\nTesting at {context_size:,} chars ({context_size // 10:,} pairs)...")

    eval_gen = MultiHopEval()
    table = AnalyticalHashTable()

    correct = 0
    start = time.time()

    for i in range(num_samples):
        sample = eval_gen.make_one(
            n_chars_problem=context_size,
            num_queries=1,
            hops=1,
            hash_pair_str_length=4,
            chain_of_thought=False,
        )

        keys, values = parse_context(sample.prompt)

        for query, expected in sample.targets.items():
            result, confidence = table.lookup(query, keys, values)
            if result == expected:
                correct += 1
            break

    elapsed = time.time() - start
    accuracy = correct / num_samples * 100
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{num_samples})")
    print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples*1000:.1f}ms per sample)")

    return accuracy


if __name__ == "__main__":
    print("="*70)
    print("ANALYTICAL HASH TABLE - Upper Bound Test")
    print("="*70)
    print("\nThis uses exact character matching (no learning).")
    print("Should achieve 100% at all scales if parsing is correct.\n")

    results = {}

    # Test at increasing scales
    test_sizes = [
        1_000,        # 100 pairs
        10_000,       # 1K pairs
        100_000,      # 10K pairs
        1_000_000,    # 100K pairs
        10_000_000,   # 1M pairs (10M chars)
    ]

    for size in test_sizes:
        # Fewer samples for larger sizes to keep runtime reasonable
        if size <= 100_000:
            num_samples = 100
        elif size <= 1_000_000:
            num_samples = 50
        else:
            num_samples = 20

        acc = test_scale(size, num_samples)
        results[size] = acc

    print("\n" + "="*70)
    print("SUMMARY: Analytical (Exact Match) Hash Table")
    print("="*70)
    for ctx, acc in sorted(results.items()):
        pairs = ctx // 10
        print(f"  {ctx:>12,} chars ({pairs:>8,} pairs): {acc:.1f}%")
