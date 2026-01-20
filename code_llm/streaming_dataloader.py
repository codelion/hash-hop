"""Streaming data loader for training on large token files.

Memory-mapped file access for efficient streaming without loading everything into memory.
"""

import mmap
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Iterator
import random


class StreamingTokenDataset:
    """Memory-mapped streaming dataset for pre-tokenized data.

    Uses mmap to access tokens on-demand without loading entire file into memory.
    Supports random access for batch sampling.
    """

    def __init__(self, data_path: str):
        """Initialize the streaming dataset.

        Args:
            data_path: Path to binary token file (format: 8-byte count + 4-byte tokens)
        """
        self.data_path = Path(data_path)
        self.file = None
        self.mmap = None
        self._num_tokens = None
        self._header_size = 8  # 8 bytes for token count (uint64)
        self._token_size = 4   # 4 bytes per token (uint32)

        self._open()

    def _open(self):
        """Open file and create memory map."""
        self.file = open(self.data_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # Read token count from header
        self._num_tokens = struct.unpack('Q', self.mmap[:8])[0]

    def __len__(self) -> int:
        """Return total number of tokens."""
        return self._num_tokens

    def __getitem__(self, idx: int) -> int:
        """Get a single token by index."""
        if idx < 0 or idx >= self._num_tokens:
            raise IndexError(f"Token index {idx} out of range [0, {self._num_tokens})")

        offset = self._header_size + idx * self._token_size
        return struct.unpack('I', self.mmap[offset:offset + 4])[0]

    def get_sequence(self, start: int, length: int) -> np.ndarray:
        """Get a sequence of tokens starting at given index.

        Args:
            start: Starting token index
            length: Number of tokens to retrieve

        Returns:
            numpy array of token IDs
        """
        if start < 0:
            start = 0
        if start + length > self._num_tokens:
            length = self._num_tokens - start

        offset = self._header_size + start * self._token_size
        end_offset = offset + length * self._token_size

        # Read bytes and convert to numpy array efficiently
        token_bytes = self.mmap[offset:end_offset]
        tokens = np.frombuffer(token_bytes, dtype=np.uint32)
        return tokens.astype(np.int32)

    def close(self):
        """Close the memory map and file."""
        if self.mmap is not None:
            try:
                self.mmap.close()
            except (ValueError, BufferError):
                pass
            self.mmap = None
        if self.file is not None:
            try:
                self.file.close()
            except (ValueError, OSError):
                pass
            self.file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StreamingBatchLoader:
    """Efficient batch loader with streaming data access.

    Generates batches on-demand without loading full dataset into memory.
    """

    def __init__(
        self,
        dataset: StreamingTokenDataset,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the batch loader.

        Args:
            dataset: StreamingTokenDataset instance
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            shuffle: Whether to randomly sample starting positions
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Calculate valid range for sequence starts
        # Need seq_len + 1 tokens for input/target pair
        self.max_start = len(dataset) - seq_len - 1
        if self.max_start <= 0:
            raise ValueError(
                f"Dataset too small ({len(dataset)} tokens) for seq_len={seq_len}. "
                f"Need at least {seq_len + 2} tokens."
            )

        # For sequential iteration
        self._current_pos = 0

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single batch of input/target pairs.

        Returns:
            inputs: (batch_size, seq_len) array of input tokens
            targets: (batch_size, seq_len) array of target tokens (shifted by 1)
        """
        inputs = []
        targets = []

        if self.shuffle:
            # Random starting positions
            starts = np.random.randint(0, self.max_start, size=self.batch_size)
        else:
            # Sequential positions with wraparound
            starts = []
            for _ in range(self.batch_size):
                starts.append(self._current_pos)
                self._current_pos += self.seq_len
                if self._current_pos >= self.max_start:
                    self._current_pos = 0
            starts = np.array(starts)

        for start in starts:
            # Get seq_len + 1 tokens: [start, start + seq_len + 1)
            tokens = self.dataset.get_sequence(start, self.seq_len + 1)
            inputs.append(tokens[:-1])   # First seq_len tokens
            targets.append(tokens[1:])   # Last seq_len tokens (shifted by 1)

        return np.stack(inputs), np.stack(targets)

    def iter_batches(self, num_batches: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches.

        Args:
            num_batches: Number of batches to generate (None for infinite)

        Yields:
            (inputs, targets) tuples
        """
        count = 0
        while num_batches is None or count < num_batches:
            yield self.get_batch()
            count += 1

    def estimate_epoch_batches(self) -> int:
        """Estimate number of batches per epoch (full pass through data)."""
        tokens_per_batch = self.batch_size * self.seq_len
        return max(1, len(self.dataset) // tokens_per_batch)


def test_streaming_loader():
    """Test the streaming data loader."""
    import tempfile
    import os

    print("Testing StreamingTokenDataset...")

    # Create test data
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        test_path = f.name

        # Write header (number of tokens)
        num_tokens = 10000
        f.write(struct.pack('Q', num_tokens))

        # Write tokens (0, 1, 2, ..., 9999)
        for i in range(num_tokens):
            f.write(struct.pack('I', i % 1000))  # Wrap at 1000 for realistic vocab

    try:
        # Test dataset
        dataset = StreamingTokenDataset(test_path)
        print(f"  Dataset size: {len(dataset)} tokens")
        assert len(dataset) == num_tokens

        # Test single access
        assert dataset[0] == 0
        assert dataset[999] == 999
        assert dataset[1000] == 0  # Wrapped

        # Test sequence access
        seq = dataset.get_sequence(0, 10)
        assert len(seq) == 10
        assert list(seq) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Test batch loader
        loader = StreamingBatchLoader(dataset, batch_size=4, seq_len=128)
        inputs, targets = loader.get_batch()

        assert inputs.shape == (4, 128)
        assert targets.shape == (4, 128)

        # Verify input/target relationship (targets shifted by 1)
        for i in range(4):
            for j in range(127):
                # Since we used (idx % 1000) pattern, check consistency
                expected_diff = (targets[i, j] - inputs[i, j]) % 1000
                assert expected_diff == 1 or (inputs[i, j] == 999 and targets[i, j] == 0)

        print("  Batch shape test passed")

        # Test iteration
        batch_count = 0
        for inputs, targets in loader.iter_batches(num_batches=10):
            batch_count += 1
            assert inputs.shape == (4, 128)
        assert batch_count == 10
        print("  Iteration test passed")

        dataset.close()
        print("All tests passed!")

    finally:
        os.unlink(test_path)


if __name__ == "__main__":
    test_streaming_loader()
