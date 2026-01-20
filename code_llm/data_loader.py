"""Data loading utilities for Python code training.

Uses The Stack dataset from Hugging Face for high-quality Python code.
"""

from datasets import load_dataset
from pathlib import Path
import json
from typing import Iterator, Optional
import random


def stream_python_code(
    num_samples: int = 10000,
    min_length: int = 100,
    max_length: int = 10000,
    seed: int = 42,
) -> Iterator[str]:
    """Stream Python code samples from The Stack.

    Args:
        num_samples: Number of samples to yield
        min_length: Minimum code length in characters
        max_length: Maximum code length in characters
        seed: Random seed for shuffling

    Yields:
        Python code strings
    """
    print("Loading The Stack (Python subset)...")

    # Load streaming dataset to avoid downloading everything
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    # Shuffle with buffer
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    count = 0
    for sample in dataset:
        code = sample.get("content", "")

        # Filter by length
        if len(code) < min_length or len(code) > max_length:
            continue

        # Basic quality filters
        # Skip if too many non-ASCII characters
        non_ascii = sum(1 for c in code if ord(c) > 127)
        if non_ascii / len(code) > 0.1:
            continue

        # Skip if it looks like auto-generated
        if "DO NOT EDIT" in code or "auto-generated" in code.lower():
            continue

        yield code
        count += 1

        if count >= num_samples:
            break

        if count % 1000 == 0:
            print(f"  Loaded {count} samples...")

    print(f"  Total: {count} samples")


def download_python_samples(
    output_dir: str = "data/python_samples",
    num_samples: int = 10000,
    samples_per_file: int = 1000,
):
    """Download Python samples and save to files.

    Args:
        output_dir: Directory to save samples
        num_samples: Total number of samples
        samples_per_file: Samples per output file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = []
    file_idx = 0

    for code in stream_python_code(num_samples):
        samples.append(code)

        if len(samples) >= samples_per_file:
            # Save batch
            output_file = output_path / f"python_{file_idx:04d}.json"
            with open(output_file, "w") as f:
                json.dump(samples, f)
            print(f"Saved {output_file}")
            samples = []
            file_idx += 1

    # Save remaining
    if samples:
        output_file = output_path / f"python_{file_idx:04d}.json"
        with open(output_file, "w") as f:
            json.dump(samples, f)
        print(f"Saved {output_file}")


def load_local_samples(data_dir: str = "data/python_samples") -> Iterator[str]:
    """Load samples from local JSON files.

    Args:
        data_dir: Directory containing JSON sample files

    Yields:
        Python code strings
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for json_file in sorted(data_path.glob("*.json")):
        with open(json_file) as f:
            samples = json.load(f)
        for sample in samples:
            yield sample


def create_training_data(
    output_path: str = "data/train_tokens.bin",
    num_samples: int = 10000,
    tokenizer=None,
):
    """Create pre-tokenized training data.

    Args:
        output_path: Path to save binary token file
        num_samples: Number of code samples to process
        tokenizer: Tokenizer instance
    """
    import struct

    if tokenizer is None:
        from tokenizer import CodeTokenizer
        tokenizer = CodeTokenizer()

    all_tokens = []

    print("Tokenizing Python code...")
    for i, code in enumerate(stream_python_code(num_samples)):
        tokens = tokenizer.encode(code)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.token_to_id.get('<NEWLINE>', 6))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} samples, {len(all_tokens):,} tokens")

    print(f"\nTotal tokens: {len(all_tokens):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size()}")

    # Save as binary
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "wb") as f:
        # Header: number of tokens
        f.write(struct.pack("Q", len(all_tokens)))
        # Tokens as 4-byte integers
        for token in all_tokens:
            f.write(struct.pack("I", token))

    print(f"Saved to {output_path}")

    # Also save tokenizer
    tokenizer_path = output.parent / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")

    return all_tokens, tokenizer


def load_training_data(data_path: str = "data/train_tokens.bin"):
    """Load pre-tokenized training data.

    Args:
        data_path: Path to binary token file

    Returns:
        List of token IDs
    """
    import struct

    with open(data_path, "rb") as f:
        # Read header
        num_tokens = struct.unpack("Q", f.read(8))[0]
        # Read tokens
        tokens = []
        for _ in range(num_tokens):
            token = struct.unpack("I", f.read(4))[0]
            tokens.append(token)

    return tokens


# Alternative: use a smaller, easier dataset for initial testing
def get_codeparrot_sample(num_samples: int = 5000) -> Iterator[str]:
    """Get samples from CodeParrot (smaller, Python-only dataset).

    This is easier to access than The Stack and good for testing.
    """
    print("Loading CodeParrot dataset...")

    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )

    count = 0
    for sample in dataset:
        code = sample.get("content", "")

        if len(code) < 100 or len(code) > 10000:
            continue

        yield code
        count += 1

        if count >= num_samples:
            break

        if count % 500 == 0:
            print(f"  Loaded {count} samples...")

    print(f"  Total: {count} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Python training data")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/train_tokens.bin", help="Output path")
    parser.add_argument("--source", type=str, default="codeparrot", choices=["stack", "codeparrot"])
    args = parser.parse_args()

    print(f"Downloading {args.num_samples} Python code samples...")

    if args.source == "codeparrot":
        # Use CodeParrot for easier access
        from tokenizer import CodeTokenizer
        tokenizer = CodeTokenizer()

        all_tokens = []
        for i, code in enumerate(get_codeparrot_sample(args.num_samples)):
            tokens = tokenizer.encode(code)
            all_tokens.extend(tokens)
            all_tokens.append(6)  # NEWLINE separator

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} samples, {len(all_tokens):,} tokens")

        # Save
        import struct
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "wb") as f:
            f.write(struct.pack("Q", len(all_tokens)))
            for token in all_tokens:
                f.write(struct.pack("I", token))

        tokenizer.save(str(output.parent / "tokenizer.json"))

        print(f"\nTotal tokens: {len(all_tokens):,}")
        print(f"Vocabulary size: {tokenizer.vocab_size()}")
        print(f"Saved to {args.output}")

    else:
        create_training_data(args.output, args.num_samples)
