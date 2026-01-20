"""Test tokenized approach with realistic program variable names.

This tests whether the approach works with:
1. Similar prefixes (user_id, user_name, user_email)
2. Common patterns (get_*, set_*, is_*)
3. Short names (i, j, x, y)
4. Mixed case (userId, UserID, user_id)
"""

import numpy as np
from typing import Dict, List, Tuple
import time
import random
import string


class HashTokenizer:
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


class FastTokenizedRetriever:
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        self.embeddings = {}

    def _get_embedding(self, tid: int) -> np.ndarray:
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
        q_emb = self._get_embedding(query_id)
        k_embs = np.array([self._get_embedding(int(k)) for k in key_ids])
        v_embs = np.array([self._get_embedding(int(v)) for v in value_ids])

        scores = k_embs @ q_emb / temperature
        scores = scores - np.max(scores)
        attn = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-10)

        out_emb = attn @ v_embs

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


def random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


def generate_realistic_pairs(n_pairs: int, pattern: str = "mixed") -> List[Tuple[str, str]]:
    """Generate realistic key-value pairs with UNIQUE values."""
    pairs = []

    if pattern == "similar_prefix":
        # Keys with similar prefixes, but ALL values are unique
        prefixes = ["user", "account", "order", "product", "customer", "payment",
                    "invoice", "session", "request", "response", "config", "cache"]
        suffixes = ["_id", "_name", "_email", "_status", "_type", "_value",
                    "_count", "_data", "_info", "_meta", "_key", "_hash"]
        used_keys = set()
        for i in range(n_pairs):
            # Generate unique key
            while True:
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                idx = random.randint(0, 999)
                key = f"{prefix}{suffix}_{idx}"
                if key not in used_keys:
                    used_keys.add(key)
                    break
            # Value is always unique
            value = random_string(16)
            pairs.append((key, value))

    elif pattern == "camelCase":
        # CamelCase variations with unique values
        words = ["user", "name", "email", "status", "type", "value",
                 "count", "id", "data", "info", "meta", "key"]
        used_keys = set()
        for i in range(n_pairs):
            while True:
                w1 = random.choice(words)
                w2 = random.choice(words)
                idx = random.randint(0, 999)
                key = f"{w1}{w2.capitalize()}{idx}"
                if key not in used_keys:
                    used_keys.add(key)
                    break
            value = random_string(16)
            pairs.append((key, value))

    elif pattern == "short":
        # Short variable names with unique values
        for i in range(n_pairs):
            key = f"v{i}"
            value = f"r{i}_{random_string(8)}"
            pairs.append((key, value))

    else:  # random (like HashHop)
        # Random strings like HashHop
        for i in range(n_pairs):
            key = random_string(16)
            value = random_string(16)
            pairs.append((key, value))

    return pairs[:n_pairs]


def test_pattern(pattern: str, sizes: List[int]):
    """Test retrieval accuracy for a given pattern."""
    print(f"\n{'='*60}")
    print(f"Testing pattern: {pattern}")
    print(f"{'='*60}")

    for n_pairs in sizes:
        model = FastTokenizedRetriever(d_model=256)
        tokenizer = HashTokenizer()

        pairs = generate_realistic_pairs(n_pairs, pattern)

        # Tokenize all pairs
        key_ids = np.array([tokenizer.encode(k) for k, v in pairs])
        val_ids = np.array([tokenizer.encode(v) for k, v in pairs])

        # Test retrieval
        correct = 0
        for i, (key, expected_value) in enumerate(pairs):
            query_id = tokenizer.encode(key)
            pred_id, attn = model.retrieve(query_id, key_ids, val_ids)
            pred_value = tokenizer.decode(pred_id)

            if pred_value == expected_value:
                correct += 1

        accuracy = correct / n_pairs * 100
        print(f"  {n_pairs:>6} pairs: {accuracy:.0f}% accuracy")


def main():
    sizes = [10, 100, 1000, 10000]

    print("="*60)
    print("REALISTIC STRING TOKENIZATION TEST (Fixed)")
    print("="*60)
    print("\nThis tests whether tokenized lookup works with realistic")
    print("variable names. ALL values are unique (no duplicates).")

    patterns = ["similar_prefix", "camelCase", "short", "random"]

    for pattern in patterns:
        test_pattern(pattern, sizes)

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)


if __name__ == "__main__":
    main()
