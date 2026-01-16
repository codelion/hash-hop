"""Trainer for Indexed Memory Transformer with autoregressive generation."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from imt.config import IMTConfig, TrainingConfig
from imt.data.chunked_dataset import ChunkedHashHopDataset, ChunkedSample
from imt.data.tokenizer import HashTokenizer
from imt.model.imt import IndexedMemoryTransformer
from imt.training.loss import (
    compute_accuracy,
    compute_contrastive_embedding_loss,
    compute_retrieval_recall,
    compute_total_loss_with_copy,
)


class IMTTrainer:
    """Trainer for Indexed Memory Transformer with autoregressive generation.

    Handles:
    - Training loop with MLX optimizers
    - Gradient checkpointing for memory efficiency
    - Teacher forcing for autoregressive generation
    - Copy mechanism supervision
    - Periodic evaluation with autoregressive generation
    - Checkpointing
    """

    def __init__(
        self,
        model: IndexedMemoryTransformer,
        config: IMTConfig,
        train_config: TrainingConfig,
        tokenizer: HashTokenizer,
        output_dir: Optional[str] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: The IMT model to train.
            config: Model configuration.
            train_config: Training configuration.
            tokenizer: Character-level tokenizer.
            output_dir: Output directory for checkpoints.
        """
        self.model = model
        self.config = config
        self.train_config = train_config
        self.tokenizer = tokenizer

        # Set memory limit to restrict VRAM usage
        if config.memory_limit_fraction < 1.0:
            self._setup_memory_limit(config.memory_limit_fraction)

        # Set output directory
        if output_dir is None:
            output_dir = train_config.output_dir
        self.output_dir = Path(output_dir) / train_config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dataset
        self.dataset = ChunkedHashHopDataset(config, train_config, tokenizer)

        # Optimizer with warmup schedule
        self.optimizer = optim.AdamW(
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Training state
        self.global_step = 0
        self.best_eval_acc = 0.0
        self.training_history: List[Dict[str, Any]] = []

    def _setup_memory_limit(self, fraction: float) -> None:
        """Configure MLX memory limit.

        Args:
            fraction: Fraction of total memory to use (0.0-1.0).
        """
        import mlx.core as mx

        # Get device memory info
        try:
            # Set memory limit using MLX's metal backend
            # MLX uses a memory pool that we can limit
            memory_info = mx.metal.device_info()
            total_memory = memory_info.get("memory_size", 0)

            if total_memory > 0:
                limit_bytes = int(total_memory * fraction)
                # Use new API if available, fall back to deprecated one
                if hasattr(mx, "set_memory_limit"):
                    mx.set_memory_limit(limit_bytes)
                else:
                    mx.metal.set_memory_limit(limit_bytes)
                print(
                    f"Memory limit set to {fraction*100:.0f}% "
                    f"({limit_bytes / (1024**3):.1f} GB of {total_memory / (1024**3):.1f} GB)"
                )
            else:
                print("Warning: Could not determine device memory, using default limits")
        except AttributeError:
            # Older MLX versions may not have these functions
            print(
                f"Warning: MLX memory limiting not available in this version. "
                f"Using reduced batch sizes instead."
            )
            # Reduce batch size as fallback
            self.config.chunk_batch_size = max(8, self.config.chunk_batch_size // 2)
            print(f"Reduced chunk_batch_size to {self.config.chunk_batch_size}")

    def _loss_fn(
        self,
        model: IndexedMemoryTransformer,
        sample: ChunkedSample,
    ) -> tuple:
        """Compute loss for a single sample using teacher forcing.

        Args:
            model: The model (passed for gradient computation).
            sample: Training sample.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Forward pass with teacher forcing
        logits, copy_attention, copy_gate, retrieval_scores, chunk_indices, index_keys, all_chunk_scores = model(
            sample.chunk_tokens,
            sample.query_tokens,
            sample.target_tokens,
        )

        # Compute loss with copy mechanism and copy attention supervision
        lambda_copy = getattr(self.train_config, 'lambda_copy', 0.1)
        lambda_copy_attn = getattr(self.train_config, 'lambda_copy_attn', 1.0)
        loss, metrics = compute_total_loss_with_copy(
            logits=logits,
            targets=sample.target_tokens,
            copy_gate=copy_gate,
            copy_attention=copy_attention,
            copy_targets=sample.copy_targets,
            retrieval_scores=retrieval_scores,
            chunk_indices=chunk_indices,
            target_chunk_indices=sample.target_chunk_indices,
            index_keys=index_keys,
            top_k=self.config.retrieval_top_k,
            pad_id=self.tokenizer.pad_id,
            lambda_retrieval=self.train_config.lambda_retrieval,
            lambda_reg=self.train_config.lambda_regularization,
            lambda_copy=lambda_copy,
            lambda_copy_attn=lambda_copy_attn,
            all_chunk_scores=all_chunk_scores,
        )

        # Add contrastive embedding loss to bootstrap retrieval learning
        # This uses the shared embeddings to create a direct supervision signal
        contrastive_loss = compute_contrastive_embedding_loss(
            query_tokens=sample.query_tokens,
            chunk_tokens=sample.chunk_tokens,
            target_chunk_indices=sample.target_chunk_indices,
            shared_embedding=model.shared_token_embed.weight,
            temperature=0.1,
        )
        lambda_contrastive = getattr(self.train_config, 'lambda_contrastive', 1.0)
        loss = loss + lambda_contrastive * contrastive_loss
        metrics["contrastive_loss"] = float(contrastive_loss)

        # Compute additional metrics
        metrics["token_accuracy"] = compute_accuracy(
            logits, sample.target_tokens, self.tokenizer.pad_id
        )
        metrics["retrieval_recall"] = compute_retrieval_recall(
            chunk_indices, sample.target_chunk_indices
        )

        return loss, metrics

    def train_step(self, sample: ChunkedSample) -> Dict[str, float]:
        """Single training step.

        Args:
            sample: Training sample.

        Returns:
            Dictionary of metrics.
        """
        # Compute loss and gradients using value_and_grad
        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)
        (loss, metrics), grads = loss_and_grad_fn(self.model, sample)

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Evaluate to sync
        mx.eval(self.model.parameters(), self.optimizer.state)

        return metrics

    def evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on held-out samples using autoregressive generation.

        Args:
            num_samples: Number of samples to evaluate on.

        Returns:
            Dictionary of evaluation metrics.
        """
        if num_samples is None:
            num_samples = self.train_config.eval_samples

        total_metrics: Dict[str, float] = {}
        exact_matches = 0
        total_queries = 0

        for _ in range(num_samples):
            sample = self.dataset.generate_sample()

            # Use autoregressive generation for evaluation (not teacher forcing)
            generated, retrieval_scores, chunk_indices = self.model.generate(
                sample.chunk_tokens,
                sample.query_tokens,
                max_length=self.config.max_hash_length,
            )

            # Also get teacher forcing logits for token accuracy comparison
            logits, _, _, _, _, _, _ = self.model(
                sample.chunk_tokens,
                sample.query_tokens,
                sample.target_tokens,
            )

            # Token-level metrics (from teacher forcing)
            acc = compute_accuracy(logits, sample.target_tokens, self.tokenizer.pad_id)
            recall = compute_retrieval_recall(chunk_indices, sample.target_chunk_indices)

            for key in ["token_accuracy", "retrieval_recall"]:
                if key not in total_metrics:
                    total_metrics[key] = 0.0

            total_metrics["token_accuracy"] += acc
            total_metrics["retrieval_recall"] += recall

            # Exact match accuracy (per query) using autoregressive generation
            for i in range(generated.shape[0]):
                # Find actual length of target (exclude padding)
                target_tokens = sample.target_tokens[i].tolist()
                actual_len = sum(1 for t in target_tokens if t != self.tokenizer.pad_id)

                # Get generated tokens (may be different length)
                gen_tokens = generated[i].tolist()

                # Decode and compare
                pred_str = self.tokenizer.decode(gen_tokens).strip()
                target_str = self.tokenizer.decode(target_tokens[:actual_len]).strip()

                if pred_str == target_str:
                    exact_matches += 1
                total_queries += 1

        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_samples

        total_metrics["exact_match_accuracy"] = exact_matches / total_queries

        return total_metrics

    def train(self, num_steps: Optional[int] = None) -> None:
        """Main training loop.

        Args:
            num_steps: Number of training steps. Uses config default if None.
        """
        if num_steps is None:
            num_steps = self.train_config.max_steps

        print(f"Starting training for {num_steps} steps...")
        print(f"Output directory: {self.output_dir}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Architecture: Autoregressive decoder with copy mechanism")

        step_times: List[float] = []
        accumulated_metrics: Dict[str, float] = {}
        accumulation_count = 0

        for step in range(num_steps):
            start_time = time.time()

            # Generate sample
            sample = self.dataset.generate_sample()

            # Training step
            metrics = self.train_step(sample)

            step_time = time.time() - start_time
            step_times.append(step_time)

            # Accumulate metrics for logging
            for key, value in metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = 0.0
                accumulated_metrics[key] += value
            accumulation_count += 1

            self.global_step += 1

            # Logging
            if step % self.train_config.log_every == 0:
                avg_time = sum(step_times[-100:]) / len(step_times[-100:])
                avg_metrics = {k: v / accumulation_count for k, v in accumulated_metrics.items()}

                print(
                    f"Step {step}: "
                    f"loss={avg_metrics.get('total_loss', 0):.4f}, "
                    f"gen={avg_metrics.get('generation_loss', 0):.4f}, "
                    f"ret={avg_metrics.get('retrieval_loss', 0):.4f}, "
                    f"copy_attn={avg_metrics.get('copy_attn_loss', 0):.4f}, "
                    f"copy_gate={avg_metrics.get('copy_gate_mean', 0):.3f}, "
                    f"acc={avg_metrics.get('token_accuracy', 0):.4f}, "
                    f"recall={avg_metrics.get('retrieval_recall', 0):.4f}, "
                    f"time={avg_time:.2f}s"
                )

                self.training_history.append(
                    {"step": step, "time": avg_time, **avg_metrics}
                )

                # Reset accumulation
                accumulated_metrics = {}
                accumulation_count = 0

            # Evaluation
            if step % self.train_config.eval_every == 0 and step > 0:
                eval_metrics = self.evaluate()
                print(
                    f"Eval at step {step}: "
                    f"exact_match={eval_metrics['exact_match_accuracy']:.4f}, "
                    f"token_acc={eval_metrics['token_accuracy']:.4f}, "
                    f"recall={eval_metrics['retrieval_recall']:.4f}"
                )

                if eval_metrics["exact_match_accuracy"] > self.best_eval_acc:
                    self.best_eval_acc = eval_metrics["exact_match_accuracy"]
                    self.save_checkpoint("best")
                    print(f"  New best model saved! Accuracy: {self.best_eval_acc:.4f}")

            # Checkpointing
            if step % self.train_config.save_every == 0 and step > 0:
                self.save_checkpoint(f"step_{step}")

        # Final save
        self.save_checkpoint("final")
        self._save_training_history()
        print(f"Training complete. Best accuracy: {self.best_eval_acc:.4f}")

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Args:
            name: Checkpoint name (e.g., "best", "step_1000").
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model weights
        # Use mlx.utils for tree flattening (API changed in newer versions)
        from mlx import utils as mlx_utils
        weights = dict(mlx_utils.tree_flatten(self.model.parameters()))
        mx.save_safetensors(str(checkpoint_dir / "weights.safetensors"), weights)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_eval_acc": self.best_eval_acc,
        }
        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save configs
        with open(checkpoint_dir / "model_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        with open(checkpoint_dir / "train_config.json", "w") as f:
            # Convert to dict, handling non-serializable fields
            train_dict = {
                k: v for k, v in self.train_config.__dict__.items() if not callable(v)
            }
            json.dump(train_dict, f, indent=2)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint directory.
        """
        checkpoint_dir = Path(path)

        # Load weights
        weights = mx.load(str(checkpoint_dir / "weights.safetensors"))
        self.model.update(weights)

        # Load state
        with open(checkpoint_dir / "state.json") as f:
            state = json.load(f)
        self.global_step = state["global_step"]
        self.best_eval_acc = state["best_eval_acc"]

        print(f"Loaded checkpoint from {checkpoint_dir}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best eval accuracy: {self.best_eval_acc:.4f}")

    def _save_training_history(self) -> None:
        """Save training history to JSON file."""
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
