"""HashHop dataset for T5 fine-tuning."""

import random
from dataclasses import dataclass
from typing import List, Tuple

from hashhop import MultiHopEval


@dataclass
class T5HashHopSample:
    """A single HashHop sample formatted for T5."""

    input_text: str  # The context with hash pairs + query
    target_text: str  # The expected answer


class HashHopT5Dataset:
    """Generate HashHop samples in T5 format.

    T5 expects text-to-text format:
    - Input: "hashhop: <context> query: <query_key>"
    - Output: "<answer>"

    The context contains hash pairs like "ABCD=EFGH" and the model
    must find the value for the given query key.
    """

    def __init__(
        self,
        n_chars_problem: int = 1000,
        num_queries: int = 1,
        hops: int = 1,
        hash_pair_str_length: int = 4,
    ):
        self.n_chars_problem = n_chars_problem
        self.num_queries = num_queries
        self.hops = hops
        self.hash_pair_str_length = hash_pair_str_length

    def generate_sample(self) -> T5HashHopSample:
        """Generate a single T5-formatted HashHop sample."""
        # Generate raw HashHop data
        sample = MultiHopEval.make_one(
            n_chars_problem=self.n_chars_problem,
            num_queries=self.num_queries,
            hops=self.hops,
            hash_pair_str_length=self.hash_pair_str_length,
            chain_of_thought=False,
        )

        # Get a random query and its answer
        query = random.choice(list(sample.targets.keys()))
        answer = sample.targets[query]

        # Format for T5
        # The prompt already uses "=" format from the original HashHop
        # Format: "hashhop: <context> query: <query>"
        input_text = f"hashhop: {sample.prompt} query: {query}"
        target_text = answer

        return T5HashHopSample(
            input_text=input_text,
            target_text=target_text,
        )

    def generate_batch(self, batch_size: int) -> List[T5HashHopSample]:
        """Generate a batch of samples."""
        return [self.generate_sample() for _ in range(batch_size)]


def test_dataset():
    """Test the dataset generation."""
    dataset = HashHopT5Dataset(n_chars_problem=500, hash_pair_str_length=4)

    sample = dataset.generate_sample()
    print("Input (first 200 chars):")
    print(sample.input_text[:200])
    print("...")
    print(f"\nInput length: {len(sample.input_text)}")
    print(f"\nTarget: {sample.target_text}")


if __name__ == "__main__":
    test_dataset()
