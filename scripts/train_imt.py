#!/usr/bin/env python3
"""Train Indexed Memory Transformer for HashHop.

Usage:
    python scripts/train_imt.py --config configs/imt_nano.yaml
    python scripts/train_imt.py --config configs/imt_nano_small.yaml --num-steps 1000
    python scripts/train_imt.py --config configs/imt_nano.yaml --resume checkpoints/imt_xxx/step_1000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.data.tokenizer import HashTokenizer
from imt.model.imt import IndexedMemoryTransformer
from imt.training.trainer import IMTTrainer


def load_config(config_path: str) -> tuple:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Tuple of (IMTConfig, TrainingConfig).
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    model_config = IMTConfig(**config_dict.get("model", {}))
    train_config = TrainingConfig(**config_dict.get("training", {}))

    return model_config, train_config


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train IMT for HashHop")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override number of training steps",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --resume)",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="Fraction of GPU memory to use (0.0-1.0). Default: 0.5",
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    model_config, train_config = load_config(args.config)

    # Print configuration summary
    print("\n" + "=" * 60)
    print("Model Configuration:")
    print("=" * 60)
    print(f"  d_model: {model_config.d_model}")
    print(f"  encoder_layers: {model_config.encoder_layers}")
    print(f"  decoder_layers: {model_config.decoder_layers}")
    print(f"  chunk_size: {model_config.chunk_size}")
    print(f"  num_clusters: {model_config.num_clusters}")
    print(f"  keys_per_chunk: {model_config.keys_per_chunk}")
    print(f"  retrieval_top_k: {model_config.retrieval_top_k}")

    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  learning_rate: {train_config.learning_rate}")
    print(f"  max_steps: {train_config.max_steps}")
    print(f"  n_chars_problem: {train_config.n_chars_problem:,}")
    print(f"  num_queries: {train_config.num_queries}")
    print(f"  hops: {train_config.hops}")
    print("=" * 60 + "\n")

    # Override memory limit if specified
    if args.memory_limit is not None:
        model_config.memory_limit_fraction = args.memory_limit
        print(f"Memory limit set to {args.memory_limit*100:.0f}%")

    # Initialize tokenizer
    tokenizer = HashTokenizer()
    model_config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Initialize model
    print("Initializing model...")
    model = IndexedMemoryTransformer(model_config)

    # Count and display parameters
    num_params = model.count_parameters()
    print(f"Total parameters: {num_params:,}")
    print(f"Estimated memory: {num_params * 4 / 1e9:.2f} GB (FP32)")

    # Create trainer
    output_dir = args.output_dir or train_config.output_dir
    trainer = IMTTrainer(
        model=model,
        config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    # Resume if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Eval only mode
    if args.eval_only:
        if not args.resume:
            print("Error: --eval-only requires --resume")
            sys.exit(1)

        print("\nRunning evaluation...")
        eval_metrics = trainer.evaluate(num_samples=50)
        print("\nEvaluation Results:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
        return

    # Train
    num_steps = args.num_steps or train_config.max_steps
    print(f"\nStarting training for {num_steps} steps...")

    try:
        trainer.train(num_steps=num_steps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
        print("Checkpoint saved.")


if __name__ == "__main__":
    main()
