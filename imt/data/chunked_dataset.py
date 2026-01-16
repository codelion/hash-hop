"""Chunked dataset for IMT training on HashHop tasks."""

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import numpy as np

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
    # Copy supervision: for each output position, which context position to copy from
    # Shape: (num_queries, max_hash_length, chunk_size)
    # For output position i, copy_targets[q, i, :] is a one-hot over chunk positions
    copy_targets: mx.array
    raw_sample: MultiHopSample  # Original sample for debugging


def make_random_string(length: int) -> str:
    """Generate a random alphanumeric string."""
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    return "".join(alphabet[ix] for ix in np.random.choice(len(alphabet), length))


class ChunkedHashHopDataset:
    """Dataset that generates chunked HashHop samples.

    Each sample contains:
    - chunk_tokens: (num_chunks, chunk_size) - tokenized chunks
    - query_tokens: (num_queries, max_hash_length) - hash queries
    - target_tokens: (num_queries, max_hash_length) - expected answers
    - target_chunk_indices: (num_queries,) - which chunk contains the answer

    Can use either original HashHop format (KEY = 'VALUE') or simplified
    format (KEY>VALUE) which is easier for the model to learn.
    """

    def __init__(
        self,
        config: IMTConfig,
        train_config: TrainingConfig,
        tokenizer: HashTokenizer,
        use_simple_format: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            config: Model configuration.
            train_config: Training configuration.
            tokenizer: Character-level tokenizer.
            use_simple_format: If True, use simplified KEY>VALUE format.
        """
        self.config = config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.use_simple_format = use_simple_format

    def _generate_simple_sample(self) -> Tuple[str, Dict[str, str]]:
        """Generate a sample with simplified KEY>VALUE format.

        Returns:
            Tuple of (prompt, targets_dict).
        """
        hash_len = self.train_config.hash_pair_str_length
        # Calculate number of pairs needed to fill n_chars_problem
        # Format: "KEY>VALUE\n" = hash_len*2 + 2 chars
        chars_per_pair = hash_len * 2 + 2
        n_pairs = max(10, self.train_config.n_chars_problem // chars_per_pair)

        # Generate random key-value pairs
        pairs: Dict[str, str] = {}
        for _ in range(n_pairs):
            key = make_random_string(hash_len)
            value = make_random_string(hash_len)
            pairs[key] = value

        # Shuffle and create prompt
        items = list(pairs.items())
        random.shuffle(items)
        lines = [f"{k}>{v}" for k, v in items]
        prompt = "\n".join(lines)

        return prompt, pairs

    def generate_sample(self) -> ChunkedSample:
        """Generate a single chunked sample.

        Returns:
            ChunkedSample with tokenized chunks, queries, and targets.
        """
        if self.use_simple_format:
            # Use simplified format
            prompt, all_targets = self._generate_simple_sample()

            # Select queries first
            all_keys = list(all_targets.keys())
            random.shuffle(all_keys)
            queries = all_keys[: self.train_config.num_queries]
            targets = [all_targets[q] for q in queries]

            # Create a sample with ONLY the selected queries/targets for debugging
            selected_targets = {q: all_targets[q] for q in queries}
            sample = MultiHopSample(
                prompt=prompt,
                completion="",
                targets=selected_targets,
            )

            # Tokenize and chunk
            chunk_tokens, line_to_chunk = self._tokenize_and_chunk(prompt)

            # Find target chunks using simple format parsing
            target_chunk_indices = self._find_target_chunks_simple(
                prompt, queries, line_to_chunk
            )
        else:
            # Use original HashHop format
            sample = MultiHopEval.make_one(
                n_chars_problem=self.train_config.n_chars_problem,
                num_queries=self.train_config.num_queries,
                hops=self.train_config.hops,
                hash_pair_str_length=self.train_config.hash_pair_str_length,
                chain_of_thought=False,
            )

            chunk_tokens, line_to_chunk = self._tokenize_and_chunk(sample.prompt)

            queries = list(sample.targets.keys())[: self.train_config.num_queries]
            targets = list(sample.targets.values())[: self.train_config.num_queries]

            target_chunk_indices = self._find_target_chunks(
                sample.prompt, queries, targets, line_to_chunk
            )

        query_tokens = self.tokenizer.encode_batch(queries, self.config.max_hash_length)
        target_tokens = self.tokenizer.encode_batch(targets, self.config.max_hash_length)

        # Compute copy targets - which positions in the target chunk contain answer
        copy_targets = self._compute_copy_targets(
            chunk_tokens, target_chunk_indices, targets
        )

        return ChunkedSample(
            chunk_tokens=chunk_tokens,
            query_tokens=query_tokens,
            target_tokens=target_tokens,
            target_chunk_indices=mx.array(target_chunk_indices, dtype=mx.int32),
            copy_targets=copy_targets,
            raw_sample=sample,
        )

    def _find_target_chunks_simple(
        self,
        prompt: str,
        queries: List[str],
        line_to_chunk: Dict[int, int],
    ) -> List[int]:
        """Find target chunks for simplified format.

        Args:
            prompt: The prompt with KEY>VALUE pairs.
            queries: List of query keys.
            line_to_chunk: Mapping from line index to chunk index.

        Returns:
            List of chunk indices for each query.
        """
        lines = prompt.split("\n")
        key_to_line: Dict[str, int] = {}
        for line_idx, line in enumerate(lines):
            if ">" in line:
                key = line.split(">")[0]
                key_to_line[key] = line_idx

        target_chunks = []
        for query in queries:
            if query in key_to_line:
                line_idx = key_to_line[query]
                chunk_idx = line_to_chunk.get(line_idx, 0)
            else:
                chunk_idx = 0  # Fallback
            target_chunks.append(chunk_idx)

        return target_chunks

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

    def _compute_copy_targets(
        self,
        chunk_tokens: mx.array,
        target_chunk_indices: List[int],
        targets: List[str],
    ) -> mx.array:
        """Compute position-specific copy supervision targets.

        For each query and each output position, create a one-hot mask indicating
        which context position contains the token to copy. This provides explicit
        supervision for where copy attention should focus at each generation step.

        Args:
            chunk_tokens: All chunk tokens (num_chunks, chunk_size).
            target_chunk_indices: List of target chunk indices per query.
            targets: List of target strings.

        Returns:
            Copy targets array of shape (num_queries, max_hash_length, chunk_size)
            where copy_targets[q, i, j] = 1.0 if output position i should copy
            from context position j.
        """
        num_queries = len(targets)
        chunk_size = chunk_tokens.shape[1]
        max_hash_length = self.config.max_hash_length

        # Initialize copy targets as numpy for easier manipulation
        # Shape: (num_queries, max_hash_length, chunk_size)
        copy_targets_np = np.zeros((num_queries, max_hash_length, chunk_size), dtype=np.float32)

        for q_idx, (chunk_idx, target) in enumerate(zip(target_chunk_indices, targets)):
            # Get the target chunk's tokens
            chunk_tokens_np = np.array(chunk_tokens[chunk_idx].tolist())

            # Encode target to get its tokens
            target_tokens = self.tokenizer.encode(target)
            target_len = len(target_tokens)

            # Find where the target sequence starts in the chunk
            start_pos = -1
            for pos in range(chunk_size - target_len + 1):
                if list(chunk_tokens_np[pos:pos + target_len]) == target_tokens:
                    start_pos = pos
                    break

            if start_pos >= 0:
                # Found the target sequence - create position-specific targets
                for i, t_tok in enumerate(target_tokens):
                    if i < max_hash_length:
                        # Output position i should copy from context position start_pos + i
                        copy_targets_np[q_idx, i, start_pos + i] = 1.0
            else:
                # Fallback: for each output position, find any matching token
                for i, t_tok in enumerate(target_tokens):
                    if i < max_hash_length and t_tok != self.tokenizer.pad_id:
                        positions = np.where(chunk_tokens_np == t_tok)[0]
                        if len(positions) > 0:
                            # Use first matching position
                            copy_targets_np[q_idx, i, positions[0]] = 1.0

        return mx.array(copy_targets_np)

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
