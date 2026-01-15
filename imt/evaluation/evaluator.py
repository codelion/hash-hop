"""Evaluation harness for IMT on HashHop tasks."""

from typing import Any, Dict, List, Optional

import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.data.chunked_dataset import ChunkedHashHopDataset
from imt.data.tokenizer import HashTokenizer
from imt.evaluation.metrics import (
    HopMetrics,
    compute_character_accuracy,
    compute_exact_match_accuracy,
    compute_hop_metrics,
    compute_retrieval_metrics,
)
from imt.model.imt import IndexedMemoryTransformer


class IMTEvaluator:
    """Comprehensive evaluation harness for IMT on HashHop.

    Evaluates:
    - Exact match accuracy
    - Character-level accuracy
    - Retrieval recall and MRR
    - Performance across different hop depths
    - Performance across different context lengths
    """

    def __init__(
        self,
        model: IndexedMemoryTransformer,
        config: IMTConfig,
        tokenizer: HashTokenizer,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Trained IMT model.
            config: Model configuration.
            tokenizer: Character-level tokenizer.
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

    def evaluate_single_sample(
        self,
        dataset: ChunkedHashHopDataset,
    ) -> Dict[str, Any]:
        """Evaluate on a single sample.

        Args:
            dataset: Dataset to generate sample from.

        Returns:
            Dictionary with per-query results.
        """
        sample = dataset.generate_sample()

        # Forward pass
        logits, retrieval_scores, chunk_indices, _ = self.model(
            sample.chunk_tokens,
            sample.query_tokens,
        )

        # Get predictions
        predictions = mx.argmax(logits, axis=-1)

        # Decode predictions and targets
        pred_strings = []
        target_strings = []

        for i in range(predictions.shape[0]):
            pred_str = self.tokenizer.strip_padding(
                self.tokenizer.decode(predictions[i].tolist())
            )
            target_str = self.tokenizer.strip_padding(
                self.tokenizer.decode(sample.target_tokens[i].tolist())
            )
            pred_strings.append(pred_str)
            target_strings.append(target_str)

        return {
            "predictions": pred_strings,
            "targets": target_strings,
            "retrieved_chunks": chunk_indices.tolist(),
            "target_chunks": sample.target_chunk_indices.tolist(),
            "retrieval_scores": retrieval_scores.tolist(),
        }

    def evaluate(
        self,
        train_config: TrainingConfig,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """Full evaluation on multiple samples.

        Args:
            train_config: Training config for data generation.
            num_samples: Number of samples to evaluate.

        Returns:
            Dictionary with aggregated metrics.
        """
        dataset = ChunkedHashHopDataset(self.config, train_config, self.tokenizer)

        all_predictions: List[str] = []
        all_targets: List[str] = []
        all_retrieved: List[List[int]] = []
        all_target_chunks: List[int] = []

        for _ in range(num_samples):
            result = self.evaluate_single_sample(dataset)
            all_predictions.extend(result["predictions"])
            all_targets.extend(result["targets"])
            all_retrieved.extend(result["retrieved_chunks"])
            all_target_chunks.extend(result["target_chunks"])

        # Compute metrics
        exact_match = compute_exact_match_accuracy(all_predictions, all_targets)
        char_accuracy = compute_character_accuracy(all_predictions, all_targets)
        retrieval_metrics = compute_retrieval_metrics(all_retrieved, all_target_chunks)

        return {
            "exact_match_accuracy": exact_match,
            "character_accuracy": char_accuracy,
            **retrieval_metrics,
            "num_samples": num_samples,
            "total_queries": len(all_predictions),
        }

    def evaluate_by_context_length(
        self,
        context_lengths: Optional[List[int]] = None,
        samples_per_length: int = 50,
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate accuracy across different context lengths.

        Args:
            context_lengths: List of context lengths to test.
            samples_per_length: Number of samples per context length.

        Returns:
            Dictionary mapping context length to metrics.
        """
        if context_lengths is None:
            context_lengths = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

        results: Dict[int, Dict[str, float]] = {}

        for length in context_lengths:
            print(f"Evaluating context length: {length:,}")

            # Create config for this context length
            train_config = TrainingConfig(
                n_chars_problem=length,
                num_queries=10,
                hops=2,
                hash_pair_str_length=16,
            )

            metrics = self.evaluate(train_config, num_samples=samples_per_length)
            metrics["n_chunks"] = length // self.config.chunk_size

            results[length] = metrics
            print(f"  Exact match: {metrics['exact_match_accuracy']:.4f}")
            print(f"  Recall@4: {metrics['recall@4']:.4f}")

        return results

    def evaluate_by_hop_depth(
        self,
        hop_depths: Optional[List[int]] = None,
        samples_per_depth: int = 50,
        context_length: int = 1_000_000,
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate accuracy across different hop depths.

        Args:
            hop_depths: List of hop depths to test.
            samples_per_depth: Number of samples per hop depth.
            context_length: Context length to use for all tests.

        Returns:
            Dictionary mapping hop depth to metrics.
        """
        if hop_depths is None:
            hop_depths = [1, 2, 3, 4]

        results: Dict[int, Dict[str, float]] = {}

        for hops in hop_depths:
            print(f"Evaluating {hops}-hop retrieval")

            train_config = TrainingConfig(
                n_chars_problem=context_length,
                num_queries=10,
                hops=hops,
                hash_pair_str_length=16,
            )

            metrics = self.evaluate(train_config, num_samples=samples_per_depth)
            metrics["hops"] = hops

            results[hops] = metrics
            print(f"  Exact match: {metrics['exact_match_accuracy']:.4f}")
            print(f"  Recall@4: {metrics['recall@4']:.4f}")

        return results

    def detailed_analysis(
        self,
        train_config: TrainingConfig,
        num_samples: int = 10,
    ) -> Dict[str, Any]:
        """Detailed analysis with per-query breakdown.

        Args:
            train_config: Training configuration.
            num_samples: Number of samples to analyze.

        Returns:
            Detailed analysis results.
        """
        dataset = ChunkedHashHopDataset(self.config, train_config, self.tokenizer)

        all_results = []
        for i in range(num_samples):
            result = self.evaluate_single_sample(dataset)

            # Add per-query analysis
            for j, (pred, target) in enumerate(
                zip(result["predictions"], result["targets"])
            ):
                all_results.append(
                    {
                        "sample_idx": i,
                        "query_idx": j,
                        "prediction": pred,
                        "target": target,
                        "correct": pred == target,
                        "retrieved_chunk": result["retrieved_chunks"][j],
                        "target_chunk": result["target_chunks"][j],
                        "chunk_retrieved": result["target_chunks"][j]
                        in result["retrieved_chunks"][j],
                    }
                )

        # Compute summary statistics
        correct = sum(1 for r in all_results if r["correct"])
        chunk_hit = sum(1 for r in all_results if r["chunk_retrieved"])

        return {
            "per_query_results": all_results,
            "summary": {
                "total_queries": len(all_results),
                "correct_predictions": correct,
                "accuracy": correct / len(all_results) if all_results else 0,
                "chunk_retrieval_hits": chunk_hit,
                "chunk_retrieval_rate": chunk_hit / len(all_results) if all_results else 0,
            },
        }
