#!/usr/bin/env python3
"""Evaluate Indexed Memory Transformer on HashHop.

Usage:
    python scripts/eval_imt.py --checkpoint checkpoints/imt_xxx/best
    python scripts/eval_imt.py --checkpoint checkpoints/imt_xxx/best --eval-type context-scaling
    python scripts/eval_imt.py --checkpoint checkpoints/imt_xxx/best --eval-type hop-depth
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.data.tokenizer import HashTokenizer
from imt.evaluation.evaluator import IMTEvaluator
from imt.model.imt import IndexedMemoryTransformer


def load_model(checkpoint_path: str) -> tuple:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.

    Returns:
        Tuple of (model, config, tokenizer).
    """
    checkpoint_dir = Path(checkpoint_path)

    # Load config
    with open(checkpoint_dir / "model_config.json") as f:
        config_dict = json.load(f)
    config = IMTConfig(**config_dict)

    # Initialize tokenizer and model
    tokenizer = HashTokenizer()
    config.vocab_size = tokenizer.vocab_size

    model = IndexedMemoryTransformer(config)

    # Load weights
    weights = mx.load(str(checkpoint_dir / "weights.safetensors"))
    model.update(weights)

    print(f"Loaded model from {checkpoint_dir}")
    print(f"Parameters: {model.count_parameters():,}")

    return model, config, tokenizer


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate IMT on HashHop")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["standard", "context-scaling", "hop-depth", "detailed"],
        default="standard",
        help="Type of evaluation to run",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1_000_000,
        help="Context length for standard/hop-depth evaluation",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Number of hops for standard evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    args = parser.parse_args()

    # Load model
    model, config, tokenizer = load_model(args.checkpoint)

    # Create evaluator
    evaluator = IMTEvaluator(model, config, tokenizer)

    # Run evaluation based on type
    if args.eval_type == "standard":
        print(f"\nRunning standard evaluation...")
        print(f"Context length: {args.context_length:,}")
        print(f"Hops: {args.hops}")
        print(f"Samples: {args.num_samples}")

        train_config = TrainingConfig(
            n_chars_problem=args.context_length,
            num_queries=10,
            hops=args.hops,
            hash_pair_str_length=16,
        )

        results = evaluator.evaluate(train_config, num_samples=args.num_samples)

        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    elif args.eval_type == "context-scaling":
        print("\nRunning context length scaling evaluation...")

        results = evaluator.evaluate_by_context_length(
            context_lengths=[100_000, 500_000, 1_000_000, 5_000_000],
            samples_per_length=args.num_samples // 4,
        )

        print("\n" + "=" * 60)
        print("Context Scaling Results:")
        print("=" * 60)
        for length, metrics in sorted(results.items()):
            print(f"\nContext: {length:,} chars ({metrics.get('n_chunks', 0):,} chunks)")
            print(f"  Exact match: {metrics['exact_match_accuracy']:.4f}")
            print(f"  Char accuracy: {metrics['character_accuracy']:.4f}")
            print(f"  Recall@4: {metrics['recall@4']:.4f}")
            print(f"  MRR: {metrics['mrr']:.4f}")

    elif args.eval_type == "hop-depth":
        print("\nRunning hop depth evaluation...")

        results = evaluator.evaluate_by_hop_depth(
            hop_depths=[1, 2, 3, 4],
            samples_per_depth=args.num_samples // 4,
            context_length=args.context_length,
        )

        print("\n" + "=" * 60)
        print("Hop Depth Results:")
        print("=" * 60)
        for hops, metrics in sorted(results.items()):
            print(f"\n{hops}-hop retrieval:")
            print(f"  Exact match: {metrics['exact_match_accuracy']:.4f}")
            print(f"  Char accuracy: {metrics['character_accuracy']:.4f}")
            print(f"  Recall@4: {metrics['recall@4']:.4f}")
            print(f"  MRR: {metrics['mrr']:.4f}")

    elif args.eval_type == "detailed":
        print("\nRunning detailed analysis...")

        train_config = TrainingConfig(
            n_chars_problem=args.context_length,
            num_queries=10,
            hops=args.hops,
            hash_pair_str_length=16,
        )

        results = evaluator.detailed_analysis(
            train_config,
            num_samples=min(args.num_samples, 10),
        )

        print("\n" + "=" * 60)
        print("Detailed Analysis Summary:")
        print("=" * 60)
        summary = results["summary"]
        print(f"  Total queries: {summary['total_queries']}")
        print(f"  Correct predictions: {summary['correct_predictions']}")
        print(f"  Accuracy: {summary['accuracy']:.4f}")
        print(f"  Chunk retrieval hits: {summary['chunk_retrieval_hits']}")
        print(f"  Chunk retrieval rate: {summary['chunk_retrieval_rate']:.4f}")

        # Show some examples
        print("\nSample results:")
        for r in results["per_query_results"][:5]:
            status = "CORRECT" if r["correct"] else "WRONG"
            print(f"  [{status}] {r['target'][:20]} -> {r['prediction'][:20]}")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
