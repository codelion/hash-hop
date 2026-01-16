#!/usr/bin/env python3
"""Debug script to compare teacher forcing vs autoregressive generation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.model.imt import IndexedMemoryTransformer
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
    import sys
    # Allow config path from command line
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/imt_benchmark_1k.yaml"
    checkpoint_name = config_path.split("/")[-1].replace(".yaml", "").replace("imt_", "")

    # Load config and model
    config, training_config = load_config(config_path)

    print(f"Loading config from {config_path}")
    print(f"  chunk_size: {config.chunk_size}")
    print(f"  retrieval_top_k: {config.retrieval_top_k}")
    print(f"  max_hash_length: {config.max_hash_length}")
    print(f"  n_chars_problem: {training_config.n_chars_problem}")

    # Create model
    model = IndexedMemoryTransformer(config)

    # Load checkpoint
    checkpoint_path = f"checkpoints/{checkpoint_name}/final"
    try:
        model.load_weights(f"{checkpoint_path}/weights.safetensors")
        print(f"Loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Create tokenizer and dataset
    tokenizer = ASCIITokenizer()
    dataset = ChunkedHashHopDataset(
        config=config,
        train_config=training_config,
        tokenizer=tokenizer,
        use_simple_format=True,
    )

    # Generate a few samples and compare
    print("\n" + "="*60)
    print("Comparing Teacher Forcing vs Autoregressive Generation")
    print("="*60)

    correct_count = 0
    total_count = 100

    for i in range(total_count):
        sample = dataset.generate_sample()

        # Get the raw sample to see the expected answer
        raw = sample.raw_sample
        query = list(raw.targets.keys())[0]
        expected_answer = raw.targets[query]

        print(f"\nSample {i+1}:")
        print(f"  Query: {query}")
        print(f"  Expected: {expected_answer}")

        # Prepare inputs
        # query_tokens and target_tokens have shape (num_queries, max_hash_length)
        # Since num_queries=1, we just use index 0 and add batch dim
        chunk_tokens = sample.chunk_tokens  # (num_chunks, chunk_size)
        query_tokens = sample.query_tokens[0:1, :]  # (1, query_len) - first query only
        target_tokens = sample.target_tokens[0:1, :]  # (1, target_len) - first target only

        # Teacher forcing pass through model
        (logits, copy_attention, copy_gate, retrieval_scores,
         chunk_indices, index_keys, all_chunk_scores) = model(
            chunk_tokens, query_tokens, target_tokens
        )

        tf_predictions = mx.argmax(logits, axis=-1)[0]
        tf_output = tokenizer.decode(tf_predictions.tolist())
        print(f"  Teacher Forcing output: {tf_output}")

        # Check retrieval
        target_idx = sample.target_chunk_indices[0].item()
        retrieved_indices = [chunk_indices[0, k].item() for k in range(config.retrieval_top_k)]
        target_in_retrieved = target_idx in retrieved_indices
        print(f"  Target chunk {target_idx} in retrieved {retrieved_indices}: {target_in_retrieved}")

        # Autoregressive generation
        generated, _, gen_chunk_indices = model.generate(
            chunk_tokens, query_tokens, max_length=config.max_hash_length
        )
        ar_output = tokenizer.decode(generated[0].tolist())
        print(f"  Autoregressive output: {ar_output}")

        # Clean outputs (remove EOS, PAD, etc)
        ar_output_clean = ar_output.replace('<EOS>', '').replace('<PAD>', '').strip()

        # Check if the first N chars match (where N = length of expected answer)
        # The model may generate more chars due to max_hash_length > hash_pair_str_length
        answer_len = len(expected_answer)
        first_n_chars = ar_output_clean[:answer_len]

        # Check if exact match (first N chars)
        if first_n_chars == expected_answer:
            correct_count += 1
            print(f"  ✓ CORRECT! (first {answer_len} chars match)")
        else:
            print(f"  ✗ WRONG (got '{first_n_chars}' vs '{expected_answer}')")

        # Show copy gate values
        copy_gate_mean = mx.mean(copy_gate).item()
        print(f"  Copy gate (mean): {copy_gate_mean:.3f}")

        # Show what tokens we're getting
        target_list = target_tokens[0].tolist()
        target_str = tokenizer.decode(target_list)
        print(f"  Target tokens decoded: {target_str}")

    print(f"\n{'='*60}")
    print(f"Exact match accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
