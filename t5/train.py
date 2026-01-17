"""Fine-tune T5-base on HashHop task."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from t5.model import T5
from t5.dataset import HashHopT5Dataset, T5HashHopSample


class T5Tokenizer:
    """Tokenizer wrapper for T5."""

    def __init__(self, model_name: str = "t5-base"):
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            legacy=False,
            model_max_length=512,
        )
        self.pad_id = self._tokenizer.pad_token_id
        self.eos_id = self._tokenizer.eos_token_id
        self.decoder_start_id = self._tokenizer.pad_token_id  # T5 uses pad as decoder start

    def encode(self, text: str, max_length: int = 512) -> mx.array:
        """Encode text to token IDs."""
        tokens = self._tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return mx.array(tokens["input_ids"][0])

    def encode_batch(
        self, texts: List[str], max_length: int = 512
    ) -> Tuple[mx.array, mx.array]:
        """Encode batch of texts, return tokens and attention mask."""
        tokens = self._tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return mx.array(tokens["input_ids"]), mx.array(tokens["attention_mask"])

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)


def load_t5_model(model_name: str = "t5-base", dtype=mx.bfloat16) -> Tuple[T5, dict]:
    """Load T5 model from HuggingFace."""
    path = Path(
        snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
    )

    with open(path / "config.json", "r") as f:
        config = SimpleNamespace(**json.load(f))

    model = T5(config)
    weights = mx.load(str(path / "model.safetensors"))
    weights = T5.sanitize(weights)
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()))

    return model, config


def compute_loss(
    model: T5,
    encoder_input: mx.array,
    decoder_input: mx.array,
    targets: mx.array,
    pad_id: int,
) -> mx.array:
    """Compute cross-entropy loss for T5.

    Args:
        model: T5 model
        encoder_input: Input token IDs (batch, seq_len)
        decoder_input: Decoder input (shifted targets) (batch, target_len)
        targets: Target token IDs (batch, target_len)
        pad_id: Padding token ID to ignore in loss
    """
    # Forward pass
    logits = model(encoder_input, decoder_input)

    # Flatten for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute log softmax
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

    # Gather log probs for target tokens
    batch_indices = mx.arange(logits_flat.shape[0])
    target_log_probs = log_probs[batch_indices, targets_flat]

    # Mask padding tokens
    mask = (targets_flat != pad_id).astype(mx.float32)
    loss = -target_log_probs * mask

    return loss.sum() / (mask.sum() + 1e-8)


def prepare_batch(
    samples: List[T5HashHopSample],
    tokenizer: T5Tokenizer,
    max_input_length: int = 512,
    max_target_length: int = 16,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Prepare a batch for training.

    Returns:
        encoder_input: Input token IDs (batch, input_len)
        decoder_input: Shifted target IDs for teacher forcing (batch, target_len)
        targets: Target token IDs (batch, target_len)
    """
    input_texts = [s.input_text for s in samples]
    target_texts = [s.target_text for s in samples]

    # Encode inputs
    encoder_input, _ = tokenizer.encode_batch(input_texts, max_length=max_input_length)

    # Encode targets
    targets, _ = tokenizer.encode_batch(target_texts, max_length=max_target_length)

    # Create decoder input (shifted right with decoder_start_id)
    batch_size = targets.shape[0]
    decoder_start = mx.full((batch_size, 1), tokenizer.decoder_start_id, dtype=mx.int32)
    decoder_input = mx.concatenate([decoder_start, targets[:, :-1]], axis=1)

    return encoder_input, decoder_input, targets


def evaluate(
    model: T5,
    tokenizer: T5Tokenizer,
    dataset: HashHopT5Dataset,
    num_samples: int = 100,
    max_input_length: int = 512,
) -> float:
    """Evaluate model on HashHop samples."""
    correct = 0

    for _ in range(num_samples):
        sample = dataset.generate_sample()

        # Encode input
        encoder_input = tokenizer.encode(sample.input_text, max_length=max_input_length)
        encoder_input = encoder_input[None, :]  # Add batch dim

        # Encode the context
        memory = model.encode(encoder_input)

        # Generate autoregressively
        decoder_input = mx.array([[tokenizer.decoder_start_id]])
        generated = []

        for _ in range(len(sample.target_text) + 2):  # +2 for safety
            logits = model.decode(decoder_input, memory)[0]
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            if next_token.item() == tokenizer.eos_id:
                break

            generated.append(next_token.item())
            decoder_input = mx.concatenate(
                [decoder_input, next_token[:, None]], axis=1
            )

        # Decode and compare
        output = tokenizer.decode(generated)
        if output.strip() == sample.target_text:
            correct += 1

    return correct / num_samples


def train(
    model_name: str = "t5-base",
    n_chars_problem: int = 1000,
    max_steps: int = 10000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    eval_every: int = 500,
    save_every: int = 2000,
    output_dir: str = "checkpoints/t5_hashhop",
    max_input_length: int = 512,
):
    """Train T5 on HashHop."""
    print(f"Loading {model_name}...")
    model, config = load_t5_model(model_name)
    tokenizer = T5Tokenizer(model_name)

    # Count parameters
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, dict):
                total += count_params(v)
        return total

    num_params = count_params(model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataset
    dataset = HashHopT5Dataset(
        n_chars_problem=n_chars_problem,
        num_queries=1,
        hops=1,
        hash_pair_str_length=4,
    )

    # Create optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    def loss_fn(params, encoder_input, decoder_input, targets):
        model.update(params)
        return compute_loss(model, encoder_input, decoder_input, targets, tokenizer.pad_id)

    # Use mx.value_and_grad directly on parameters
    loss_and_grad = mx.value_and_grad(loss_fn)

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining on {n_chars_problem} char contexts for {max_steps} steps...")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")

    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Generate batch
        samples = dataset.generate_batch(batch_size)
        encoder_input, decoder_input, targets = prepare_batch(
            samples, tokenizer, max_input_length=max_input_length
        )

        # Forward and backward
        params = model.parameters()
        loss, grads = loss_and_grad(params, encoder_input, decoder_input, targets)

        # Update
        new_params = optimizer.apply_gradients(grads, params)
        model.update(new_params)
        mx.eval(model.parameters(), optimizer.state)

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Step {step}: loss={float(loss):.4f}, "
                f"time={elapsed:.1f}s, "
                f"steps/s={step/elapsed:.2f}"
            )

        # Evaluation
        if step % eval_every == 0:
            print("\nEvaluating...")
            accuracy = evaluate(model, tokenizer, dataset, num_samples=50)
            print(f"Step {step}: accuracy={accuracy*100:.1f}%\n")

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = output_path / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)

            # Save weights - flatten nested dict/list structure
            def flatten_params(params, prefix=""):
                flat = {}
                if isinstance(params, dict):
                    for k, v in params.items():
                        key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, mx.array):
                            flat[key] = v
                        elif isinstance(v, (dict, list)):
                            flat.update(flatten_params(v, key))
                elif isinstance(params, list):
                    for i, v in enumerate(params):
                        key = f"{prefix}.{i}" if prefix else str(i)
                        if isinstance(v, mx.array):
                            flat[key] = v
                        elif isinstance(v, (dict, list)):
                            flat.update(flatten_params(v, key))
                return flat

            weights = flatten_params(model.parameters())
            mx.save_safetensors(str(ckpt_path / "weights.safetensors"), weights)

            # Save config
            with open(ckpt_path / "config.json", "w") as f:
                json.dump(vars(config), f, indent=2)

            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = output_path / "final"
    final_path.mkdir(exist_ok=True)

    def flatten_params_final(params, prefix=""):
        flat = {}
        if isinstance(params, dict):
            for k, v in params.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, mx.array):
                    flat[key] = v
                elif isinstance(v, (dict, list)):
                    flat.update(flatten_params_final(v, key))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                key = f"{prefix}.{i}" if prefix else str(i)
                if isinstance(v, mx.array):
                    flat[key] = v
                elif isinstance(v, (dict, list)):
                    flat.update(flatten_params_final(v, key))
        return flat

    weights = flatten_params_final(model.parameters())
    mx.save_safetensors(str(final_path / "weights.safetensors"), weights)
    with open(final_path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nTraining complete. Model saved to {final_path}")

    # Final evaluation
    print("\nFinal evaluation...")
    accuracy = evaluate(model, tokenizer, dataset, num_samples=100)
    print(f"Final accuracy: {accuracy*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 on HashHop")
    parser.add_argument("--model", type=str, default="t5-base", help="T5 model name")
    parser.add_argument(
        "--context-size", type=int, default=1000, help="Context size in characters"
    )
    parser.add_argument("--max-steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=500, help="Eval frequency")
    parser.add_argument("--save-every", type=int, default=2000, help="Save frequency")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/t5_hashhop", help="Output dir"
    )
    parser.add_argument(
        "--max-input-length", type=int, default=512, help="Max input token length"
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        n_chars_problem=args.context_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        max_input_length=args.max_input_length,
    )
