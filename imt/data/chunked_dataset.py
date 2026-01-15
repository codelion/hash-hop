"""Chunked dataset for IMT training on HashHop tasks."""

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlx.core as mx

from hashhop.generate import MultiHopEval, MultiHopSample
from imt.config import IMTConfig, TrainingConfig
from imt.data.tokenizer import HashTokenizer


@dataclass
class ChunkedSample:
    """A chunked HashHop sample for IMT training."""

    chunk_tokens: mx.array  # (num_chunks, chunk_size)
    query_tokens: mx.array  # (num_queries, max_hash_length)
    target_tokens: mx.array  # (num_queries, max_hash_length)
    target_chunk_indices: mx.array  # (num_queries,) - which chunk contains answer
    raw_sample: MultiHopSample  # Original sample for debugging


class ChunkedHashHopDataset:
    """Dataset that generates chunked HashHop samples.

    Each sample contains:
    - chunk_tokens: (num_chunks, chunk_size) - tokenized chunks
    - query_tokens: (num_queries, max_hash_length) - hash queries
    - target_tokens: (num_queries, max_hash_length) - expected answers
    - target_chunk_indices: (num_queries,) - which chunk contains the answer
    """

    def __init__(
        self,
        config: IMTConfig,
        train_config: TrainingConfig,
        tokenizer: HashTokenizer,
    ) -> None:
        """Initialize dataset.

        Args:
            config: Model configuration.
            train_config: Training configuration.
            tokenizer: Character-level tokenizer.
        """
        self.config = config
        self.train_config = train_config
        self.tokenizer = tokenizer

    def generate_sample(self) -> ChunkedSample:
        """Generate a single chunked sample.

        Returns:
            ChunkedSample with tokenized chunks, queries, and targets.
        """
        # Generate raw HashHop sample
        sample = MultiHopEval.make_one(
            n_chars_problem=self.train_config.n_chars_problem,
            num_queries=self.train_config.num_queries,
            hops=self.train_config.hops,
            hash_pair_str_length=self.train_config.hash_pair_str_length,
            chain_of_thought=False,  # Direct mapping for IMT
        )

        # Tokenize and chunk the prompt
        chunk_tokens, line_to_chunk = self._tokenize_and_chunk(sample.prompt)

        # Prepare queries and targets (only use num_queries, as per the completion)
        queries = list(sample.targets.keys())[: self.train_config.num_queries]
        targets = list(sample.targets.values())[: self.train_config.num_queries]

        query_tokens = self.tokenizer.encode_batch(queries, self.config.max_hash_length)
        target_tokens = self.tokenizer.encode_batch(targets, self.config.max_hash_length)

        # Find which chunks contain the target answers (for retrieval supervision)
        target_chunk_indices = self._find_target_chunks(
            sample.prompt, queries, targets, line_to_chunk
        )

        return ChunkedSample(
            chunk_tokens=chunk_tokens,
            query_tokens=query_tokens,
            target_tokens=target_tokens,
            target_chunk_indices=mx.array(target_chunk_indices, dtype=mx.int32),
            raw_sample=sample,
        )

    def _tokenize_and_chunk(self, prompt: str) -> Tuple[mx.array, Dict[int, int]]:
        """Tokenize prompt and split into fixed-size chunks.

        Args:
            prompt: The full HashHop prompt with hash pairs.

        Returns:
            chunk_tokens: Array of shape (num_chunks, chunk_size).
            line_to_chunk: Mapping from line index to chunk index.
        """
        # Tokenize entire prompt
        token_ids = self.tokenizer.encode(prompt)

        # Split into chunks
        chunk_size = self.config.chunk_size
        num_tokens = len(token_ids)
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size

        # Pad to fill last chunk
        padded_length = num_chunks * chunk_size
        token_ids = token_ids + [self.tokenizer.pad_id] * (padded_length - num_tokens)

        # Reshape into chunks
        chunk_tokens = mx.array(token_ids, dtype=mx.int32).reshape(num_chunks, chunk_size)

        # Build line-to-chunk mapping
        line_to_chunk: Dict[int, int] = {}
        lines = prompt.split("\n")
        char_offset = 0
        for line_idx, line in enumerate(lines):
            # Character position maps directly to token position for char-level tokenizer
            token_pos = char_offset
            chunk_idx = token_pos // chunk_size
            line_to_chunk[line_idx] = min(chunk_idx, num_chunks - 1)
            char_offset += len(line) + 1  # +1 for newline

        return chunk_tokens, line_to_chunk

    def _find_target_chunks(
        self,
        prompt: str,
        queries: List[str],
        targets: List[str],
        line_to_chunk: Dict[int, int],
    ) -> List[int]:
        """Find which chunks contain each target answer.

        For multi-hop queries, we need to find the chunk containing the
        hash pair where the query's final target appears as a value.

        Args:
            prompt: The full HashHop prompt.
            queries: List of query hash strings.
            targets: List of target hash strings.
            line_to_chunk: Mapping from line index to chunk index.

        Returns:
            List of chunk indices for each query.
        """
        lines = prompt.split("\n")
        target_chunks = []

        # Build a mapping from hash value to line index
        value_to_line: Dict[str, int] = {}
        for line_idx, line in enumerate(lines):
            if " = " in line:
                # Parse the hash pair: "KEY = VALUE" or "KEY = 'VALUE'"
                parts = line.split(" = ")
                if len(parts) == 2:
                    value = parts[1].strip("'")
                    value_to_line[value] = line_idx

        # For each query, find the chunk containing its target
        for query, target in zip(queries, targets):
            if target in value_to_line:
                line_idx = value_to_line[target]
                chunk_idx = line_to_chunk.get(line_idx, 0)
            else:
                # Fallback: search for target in any line
                chunk_idx = self._search_for_target(lines, target, line_to_chunk)

            target_chunks.append(chunk_idx)

        return target_chunks

    def _search_for_target(
        self,
        lines: List[str],
        target: str,
        line_to_chunk: Dict[int, int],
    ) -> int:
        """Search for a target hash in the lines.

        Args:
            lines: List of lines from the prompt.
            target: Target hash string to find.
            line_to_chunk: Mapping from line index to chunk index.

        Returns:
            Chunk index containing the target, or 0 if not found.
        """
        for line_idx, line in enumerate(lines):
            if f"= '{target}'" in line or f"= {target}" in line:
                return line_to_chunk.get(line_idx, 0)
        return 0

    def stream_samples(
        self,
        num_samples: int,
    ) -> Generator[ChunkedSample, None, None]:
        """Stream samples one at a time for memory efficiency.

        Args:
            num_samples: Number of samples to generate.

        Yields:
            ChunkedSample instances.
        """
        for _ in range(num_samples):
            yield self.generate_sample()

    def generate_batch(self, batch_size: int) -> List[ChunkedSample]:
        """Generate a batch of samples.

        Note: Each sample may have different numbers of chunks due to
        random prompt sizes. For batch processing, consider padding
        or bucketing by chunk count.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            List of ChunkedSample instances.
        """
        return [self.generate_sample() for _ in range(batch_size)]


def collate_samples(
    samples: List[ChunkedSample],
) -> Dict[str, Any]:
    """Collate multiple samples for batch processing.

    Since samples may have different numbers of chunks, this function
    pads them to the maximum chunk count in the batch.

    Args:
        samples: List of ChunkedSample instances.

    Returns:
        Dictionary with batched tensors.
    """
    max_chunks = max(s.chunk_tokens.shape[0] for s in samples)
    chunk_size = samples[0].chunk_tokens.shape[1]
    num_queries = samples[0].query_tokens.shape[0]
    max_hash_length = samples[0].query_tokens.shape[1]

    # Pad chunk tokens
    padded_chunks = []
    chunk_masks = []
    for s in samples:
        num_chunks = s.chunk_tokens.shape[0]
        if num_chunks < max_chunks:
            padding = mx.zeros((max_chunks - num_chunks, chunk_size), dtype=mx.int32)
            padded = mx.concatenate([s.chunk_tokens, padding], axis=0)
        else:
            padded = s.chunk_tokens
        padded_chunks.append(padded)
        # Create mask: 1 for real chunks, 0 for padding
        mask = mx.concatenate(
            [mx.ones(num_chunks), mx.zeros(max_chunks - num_chunks)]
        )
        chunk_masks.append(mask)

    return {
        "chunk_tokens": mx.stack(padded_chunks),  # (batch, max_chunks, chunk_size)
        "chunk_masks": mx.stack(chunk_masks),  # (batch, max_chunks)
        "query_tokens": mx.stack([s.query_tokens for s in samples]),
        "target_tokens": mx.stack([s.target_tokens for s in samples]),
        "target_chunk_indices": mx.stack([s.target_chunk_indices for s in samples]),
    }
