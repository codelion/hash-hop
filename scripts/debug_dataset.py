#!/usr/bin/env python3
"""Debug script to check dataset generation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.data.tokenizer import ASCIITokenizer
from imt.data.chunked_dataset import ChunkedHashHopDataset


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    model_config = IMTConfig(**config_dict.get("model", {}))
    train_config = TrainingConfig(**config_dict.get("training", {}))

    return model_config, train_config


def main():
    # Load config
    config_path = "configs/imt_autoregressive_tiny.yaml"
    config, training_config = load_config(config_path)

    # Create tokenizer and dataset
    tokenizer = ASCIITokenizer()
    dataset = ChunkedHashHopDataset(
        config=config,
        train_config=training_config,
        tokenizer=tokenizer,
        use_simple_format=True,
    )

    print("Checking dataset generation...")
    print(f"Format: {'KEY>VALUE' if True else 'KEY = VALUE'}")
    print(f"Hash length: {training_config.hash_pair_str_length}")
    print()

    for i in range(5):
        sample = dataset.generate_sample()

        # Get the raw sample
        raw = sample.raw_sample
        query = list(raw.targets.keys())[0]
        expected_answer = raw.targets[query]

        print(f"Sample {i+1}:")
        print(f"  Query: {query}")
        print(f"  Expected answer: {expected_answer}")

        # Decode the tokenized versions
        query_tokens = sample.query_tokens[0].tolist()
        target_tokens = sample.target_tokens[0].tolist()
        query_decoded = tokenizer.decode(query_tokens)
        target_decoded = tokenizer.decode(target_tokens)

        print(f"  Query tokens decoded: {query_decoded}")
        print(f"  Target tokens decoded: {target_decoded}")

        # Check the raw prompt
        print(f"  Prompt snippet: {raw.prompt[:200]}...")

        # Show target chunk info
        target_idx = sample.target_chunk_indices[0].item()
        chunk_tokens = sample.chunk_tokens[target_idx].tolist()
        chunk_decoded = tokenizer.decode(chunk_tokens)
        print(f"  Target chunk {target_idx}: {chunk_decoded}")

        # Verify match
        if target_decoded.strip() == expected_answer:
            print(f"  ✓ Target matches expected")
        else:
            print(f"  ✗ MISMATCH: target='{target_decoded}' vs expected='{expected_answer}'")

        print()


if __name__ == "__main__":
    main()
