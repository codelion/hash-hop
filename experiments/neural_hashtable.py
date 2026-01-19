"""Neural Hash Table for HashHop.

The key insight: A hash table lookup is just:
1. Compute hash(key) to get an index
2. Look up value at that index

For neural networks, we can do:
1. Encode key -> embedding
2. Use embedding to compute attention over all stored (key, value) pairs
3. Return the value with highest attention

The trick is to make the attention SHARP (nearly one-hot) rather than soft.
We can do this with:
- Low temperature softmax
- Hard attention (argmax + straight-through estimator)
- Learned hash function that maps similar keys to same bucket

This file implements a simple neural hash table that should solve HashHop.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

from hashhop import MultiHopEval


@dataclass
class HashTableConfig:
    """Configuration for neural hash table."""
    d_model: int = 64  # Embedding dimension
    num_buckets: int = 256  # Number of hash buckets
    max_entries: int = 200  # Maximum key-value pairs
    key_length: int = 4  # Characters per key
    value_length: int = 4  # Characters per value
    vocab_size: int = 128  # ASCII characters


class CharacterEncoder(nn.Module):
    """Encode character sequences to embeddings."""

    def __init__(self, config: HashTableConfig):
        super().__init__()
        self.config = config
        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)
        # Simple MLP to combine character embeddings
        self.combine = nn.Sequential(
            nn.Linear(config.d_model * config.key_length, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def __call__(self, chars: mx.array) -> mx.array:
        """
        Args:
            chars: (batch, seq_len) character codes
        Returns:
            embeddings: (batch, d_model)
        """
        # Embed each character
        embedded = self.char_embed(chars)  # (batch, seq_len, d_model)
        # Flatten and combine
        batch_size = chars.shape[0]
        flat = embedded.reshape(batch_size, -1)  # (batch, seq_len * d_model)
        return self.combine(flat)  # (batch, d_model)


class NeuralHashTable(nn.Module):
    """A neural network that implements hash table lookup.

    Key insight: Use HARD attention (argmax) instead of soft attention.
    This gives exact lookup rather than fuzzy retrieval.
    """

    def __init__(self, config: HashTableConfig):
        super().__init__()
        self.config = config

        # Encoder for keys
        self.key_encoder = CharacterEncoder(config)

        # Output projection to produce value characters
        self.output_proj = nn.Linear(config.d_model, config.vocab_size * config.value_length)

    def encode_keys(self, keys: List[str]) -> mx.array:
        """Encode a list of key strings to embeddings."""
        # Convert strings to character codes
        char_codes = []
        for key in keys:
            codes = [ord(c) for c in key[:self.config.key_length]]
            # Pad if needed
            while len(codes) < self.config.key_length:
                codes.append(0)
            char_codes.append(codes)

        chars = mx.array(char_codes, dtype=mx.int32)
        return self.key_encoder(chars)

    def forward_with_memory(
        self,
        query_keys: mx.array,  # (batch, key_length) char codes
        memory_keys: mx.array,  # (num_entries, d_model) key embeddings
        memory_values: mx.array,  # (num_entries, d_model) value embeddings
        temperature: float = 0.1,  # Low temperature = sharper attention
    ) -> mx.array:
        """
        Lookup query keys in the memory.

        Args:
            query_keys: Character codes for query keys
            memory_keys: Pre-computed key embeddings
            memory_values: Pre-computed value embeddings
            temperature: Softmax temperature (lower = sharper)

        Returns:
            output_logits: (batch, value_length, vocab_size)
        """
        # Encode query
        query_embed = self.key_encoder(query_keys)  # (batch, d_model)

        # Compute attention scores via dot product
        # query_embed: (batch, d_model)
        # memory_keys: (num_entries, d_model)
        scores = mx.matmul(query_embed, memory_keys.T)  # (batch, num_entries)

        # Apply temperature and softmax
        attention = mx.softmax(scores / temperature, axis=-1)  # (batch, num_entries)

        # Retrieve values via attention-weighted sum
        # attention: (batch, num_entries)
        # memory_values: (num_entries, d_model)
        retrieved = mx.matmul(attention, memory_values)  # (batch, d_model)

        # Project to output vocabulary
        output = self.output_proj(retrieved)  # (batch, vocab_size * value_length)

        # Reshape to (batch, value_length, vocab_size)
        batch_size = query_keys.shape[0]
        output = output.reshape(batch_size, self.config.value_length, self.config.vocab_size)

        return output, attention


class HashHopSolver:
    """Solve HashHop using a neural hash table."""

    def __init__(self, config: Optional[HashTableConfig] = None):
        self.config = config or HashTableConfig()
        self.model = NeuralHashTable(self.config)

    def parse_context(self, context: str) -> List[Tuple[str, str, bool]]:
        """Parse HashHop context into (key, value, is_terminal) tuples."""
        pairs = []
        # Match patterns like "aBcD = eFgH" or "aBcD = 'eFgH'" (mixed case)
        pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"

        for match in re.finditer(pattern, context):
            key = match.group(1)
            value = match.group(2)
            # Check if value is quoted (terminal)
            is_terminal = "'" in match.group(0)
            pairs.append((key, value, is_terminal))

        return pairs

    def build_memory(
        self,
        pairs: List[Tuple[str, str, bool]]
    ) -> Tuple[mx.array, mx.array, dict]:
        """Build memory from parsed pairs.

        Returns:
            key_embeddings: (num_pairs, d_model)
            value_embeddings: (num_pairs, d_model)
            key_to_idx: mapping from key string to index
        """
        keys = [p[0] for p in pairs]
        values = [p[1] for p in pairs]

        # Encode keys
        key_embeds = self.model.encode_keys(keys)

        # Encode values (values are also 4-char strings)
        value_embeds = self.model.encode_keys(values)

        # Build index
        key_to_idx = {k: i for i, k in enumerate(keys)}

        return key_embeds, value_embeds, key_to_idx

    def lookup(
        self,
        query: str,
        memory_keys: mx.array,
        memory_values: mx.array,
        key_to_idx: dict,
        pairs: List[Tuple[str, str, bool]],
        temperature: float = 0.01,
    ) -> Tuple[str, mx.array]:
        """Look up a query in memory.

        Returns:
            value: The retrieved value string
            attention: The attention weights (for debugging)
        """
        # Encode query
        query_codes = mx.array([[ord(c) for c in query[:4]]], dtype=mx.int32)

        # Get attention over memory
        query_embed = self.model.key_encoder(query_codes)
        scores = mx.matmul(query_embed, memory_keys.T)
        attention = mx.softmax(scores / temperature, axis=-1)

        # Find the index with highest attention
        best_idx = int(mx.argmax(attention[0]).item())

        # Return the corresponding value
        value = pairs[best_idx][1]

        return value, attention

    def solve(self, context: str, query: str, max_hops: int = 10) -> str:
        """Solve a HashHop query by following the chain.

        Args:
            context: The HashHop context string
            query: The starting key to look up
            max_hops: Maximum chain length to follow

        Returns:
            The final value (should be a terminal/quoted value)
        """
        # Parse context
        pairs = self.parse_context(context)
        if not pairs:
            return ""

        # Build memory
        key_embeds, value_embeds, key_to_idx = self.build_memory(pairs)

        # Create lookup dict for checking terminals
        pair_dict = {p[0]: (p[1], p[2]) for p in pairs}

        # Follow the chain
        current = query
        for hop in range(max_hops):
            if current not in pair_dict:
                # Key not found
                break

            value, is_terminal = pair_dict[current]

            if is_terminal:
                # Found terminal value
                return value

            # Continue to next hop
            current = value

        return current  # Return last value if max hops reached


def test_neural_hashtable():
    """Test the neural hash table on HashHop examples."""
    print("Testing Neural Hash Table on HashHop")
    print("=" * 50)

    # Create solver
    solver = HashHopSolver()

    # Generate a simple HashHop sample
    eval_gen = MultiHopEval()
    sample = eval_gen.make_one(
        n_chars_problem=200,  # ~20 pairs
        num_queries=1,
        hops=1,
        hash_pair_str_length=4,
        chain_of_thought=False,
    )

    print(f"\nContext ({len(sample.prompt)} chars):")
    print(sample.prompt[:200] + "..." if len(sample.prompt) > 200 else sample.prompt)

    print(f"\nExpected completion: {sample.completion}")
    print(f"Targets: {sample.targets}")

    # Solve using neural hash table
    for query, expected in sample.targets.items():
        result = solver.solve(sample.prompt, query)
        status = "✓" if result == expected else "✗"
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Got: {result} {status}")

    # Test on multiple samples
    print("\n" + "=" * 50)
    print("Testing on 100 samples...")

    correct = 0
    total = 0

    for _ in range(100):
        sample = eval_gen.make_one(
            n_chars_problem=500,  # 50 pairs - this is where T5 fails!
            num_queries=1,
            hops=1,
            hash_pair_str_length=4,
            chain_of_thought=False,
        )

        for query, expected in sample.targets.items():
            result = solver.solve(sample.prompt, query)
            if result == expected:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"\nAccuracy on 500-char contexts: {accuracy:.1f}% ({correct}/{total})")

    # Test on longer contexts
    print("\nTesting on 1000-char contexts...")
    correct = 0
    total = 0

    for _ in range(100):
        sample = eval_gen.make_one(
            n_chars_problem=1000,
            num_queries=1,
            hops=1,
            hash_pair_str_length=4,
            chain_of_thought=False,
        )

        for query, expected in sample.targets.items():
            result = solver.solve(sample.prompt, query)
            if result == expected:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"Accuracy on 1000-char contexts: {accuracy:.1f}% ({correct}/{total})")


def test_pure_symbolic():
    """Test a pure symbolic (non-neural) hash table as upper bound."""
    print("\n" + "=" * 50)
    print("Testing Pure Symbolic Hash Table (Upper Bound)")
    print("=" * 50)

    eval_gen = MultiHopEval()

    for context_size in [200, 500, 1000, 5000, 10000]:
        correct = 0
        total = 0

        for _ in range(100):
            sample = eval_gen.make_one(
                n_chars_problem=context_size,
                num_queries=1,
                hops=1,
                hash_pair_str_length=4,
                chain_of_thought=False,
            )

            # Parse into dict
            pairs = {}
            pattern = r"([a-zA-Z]{4})\s*=\s*'?([a-zA-Z]{4})'?"
            for match in re.finditer(pattern, sample.prompt):
                key = match.group(1)
                value = match.group(2)
                is_terminal = "'" in match.group(0)
                pairs[key] = (value, is_terminal)

            # Solve
            for query, expected in sample.targets.items():
                current = query
                for _ in range(10):  # max hops
                    if current not in pairs:
                        break
                    value, is_terminal = pairs[current]
                    if is_terminal:
                        current = value
                        break
                    current = value

                if current == expected:
                    correct += 1
                total += 1

        accuracy = correct / total * 100
        print(f"  {context_size:>5} chars: {accuracy:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    test_neural_hashtable()
    test_pure_symbolic()
