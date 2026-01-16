#!/usr/bin/env python3
"""Evaluate a specific checkpoint."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.model.imt import IndexedMemoryTransformer
from imt.data.tokenizer import ASCIITokenizer
from imt.data.chunked_dataset import ChunkedHashHopDataset


def load_config(config_path: str):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    model_config = IMTConfig(**config_dict.get("model", {}))
    train_config = TrainingConfig(**config_dict.get("training", {}))
    return model_config, train_config


def main():
    if len(sys.argv) < 3:
        print("Usage: eval_checkpoint.py <config> <checkpoint_path> [num_samples]")
        sys.exit(1)

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    config, training_config = load_config(config_path)

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluating {num_samples} samples...")

    model = IndexedMemoryTransformer(config)
    model.load_weights(f"{checkpoint_path}/weights.safetensors")

    tokenizer = ASCIITokenizer()
    dataset = ChunkedHashHopDataset(
        config=config,
        train_config=training_config,
        tokenizer=tokenizer,
        use_simple_format=True,
    )

    correct = 0
    retrieval_ok = 0

    for i in range(num_samples):
        sample = dataset.generate_sample()
        raw = sample.raw_sample
        query = list(raw.targets.keys())[0]
        expected_answer = raw.targets[query]

        chunk_tokens = sample.chunk_tokens
        query_tokens = sample.query_tokens[0:1, :]

        # Generate
        generated, _, chunk_indices = model.generate(
            chunk_tokens, query_tokens, max_length=config.max_hash_length
        )
        output = tokenizer.decode(generated[0].tolist())
        output_clean = output.replace('<EOS>', '').replace('<PAD>', '').strip()

        # Check retrieval
        target_idx = sample.target_chunk_indices[0].item()
        retrieved_indices = [chunk_indices[0, k].item() for k in range(config.retrieval_top_k)]
        if target_idx in retrieved_indices:
            retrieval_ok += 1

        # Check answer
        answer_len = len(expected_answer)
        if output_clean[:answer_len] == expected_answer:
            correct += 1

    print(f"\nResults:")
    print(f"  Exact match: {correct}/{num_samples} = {100*correct/num_samples:.1f}%")
    print(f"  Retrieval recall: {retrieval_ok}/{num_samples} = {100*retrieval_ok/num_samples:.1f}%")


if __name__ == "__main__":
    main()
