"""Training script with streaming data loader for memory efficiency.

Key improvements over mlx_train.py:
1. Memory-mapped streaming - doesn't load all tokens into memory
2. Configurable sequence length (1024-2048 recommended for code)
3. Memory-friendly batch sizes
4. Better logging and checkpointing
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from pathlib import Path
import time
import argparse
from dataclasses import dataclass
from typing import Optional

from mlx_model import create_model
from tokenizer import CodeTokenizer
from streaming_dataloader import StreamingTokenDataset, StreamingBatchLoader


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults for code LLMs."""

    # Model
    model_size: str = "small"
    max_seq_len: int = 1024  # Longer context for code

    # Training
    batch_size: int = 2      # Small batch for memory efficiency
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0

    # Gradient accumulation for effective larger batch
    grad_accum_steps: int = 4  # Effective batch = batch_size * grad_accum_steps

    # Logging
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    save_path: str = "checkpoints"

    # Data
    data_path: str = "data/train_tokens_large.bin"
    tokenizer_path: str = "data/tokenizer.json"


def loss_fn(model, inputs: mx.array, targets: mx.array) -> mx.array:
    """Compute cross-entropy loss."""
    logits = model(inputs)
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss


def train(config: TrainConfig):
    """Train with streaming data loader."""
    print("=" * 70)
    print("Code LLM Training with Streaming Data Loader")
    print("=" * 70)

    # Load tokenizer
    tokenizer = CodeTokenizer()
    tokenizer.load(config.tokenizer_path)
    vocab_size = tokenizer.vocab_size()
    print(f"\nTokenizer loaded: {vocab_size:,} tokens")

    # Initialize streaming dataset
    print(f"Loading dataset from {config.data_path}...")
    dataset = StreamingTokenDataset(config.data_path)
    print(f"  Total tokens: {len(dataset):,}")
    print(f"  Dataset size: {len(dataset) * 4 / 1e9:.2f} GB")

    # Create batch loader
    loader = StreamingBatchLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        seq_len=config.max_seq_len,
        shuffle=True,
    )
    batches_per_epoch = loader.estimate_epoch_batches()
    print(f"  Estimated batches/epoch: {batches_per_epoch:,}")

    # Create model
    model = create_model(vocab_size, config.model_size)
    n_params = model.count_params()
    print(f"\nModel: {config.model_size}")
    print(f"  Parameters: {n_params:,}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_layers: {model.n_layers}")

    # Training config summary
    effective_batch = config.batch_size * config.grad_accum_steps
    tokens_per_step = effective_batch * config.max_seq_len
    print(f"\nTraining config:")
    print(f"  Sequence length: {config.max_seq_len}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.grad_accum_steps}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Tokens per optimizer step: {tokens_per_step:,}")
    print(f"  Max steps: {config.max_steps:,}")
    print(f"  Total tokens seen: {config.max_steps * tokens_per_step:,}")

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Loss and grad function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Create checkpoint directory
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    print(f"\nStarting training for {config.max_steps} steps...")
    print("-" * 70)

    start_time = time.time()
    total_loss = 0.0
    step_loss = 0.0
    accum_count = 0
    global_step = 0
    tokens_processed = 0

    # Accumulated gradients
    accumulated_grads = None

    batch_iter = loader.iter_batches()

    while global_step < config.max_steps:
        # Get batch
        inputs_np, targets_np = next(batch_iter)
        inputs = mx.array(inputs_np)
        targets = mx.array(targets_np)

        # Forward and backward
        loss, grads = loss_and_grad(model, inputs, targets)
        step_loss += loss.item()
        accum_count += 1
        tokens_processed += config.batch_size * config.max_seq_len

        # Accumulate gradients
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mlx_utils.tree_map(
                lambda a, g: a + g, accumulated_grads, grads
            )

        # Optimizer step after accumulation
        if accum_count >= config.grad_accum_steps:
            global_step += 1

            # Average gradients
            accumulated_grads = mlx_utils.tree_map(
                lambda g: g / config.grad_accum_steps, accumulated_grads
            )

            # Learning rate schedule with warmup
            if global_step < config.warmup_steps:
                lr = config.learning_rate * global_step / config.warmup_steps
            else:
                # Cosine decay
                progress = (global_step - config.warmup_steps) / (
                    config.max_steps - config.warmup_steps
                )
                lr = config.learning_rate * 0.5 * (1.0 + mx.cos(mx.array(progress * 3.14159)).item())
            optimizer.learning_rate = lr

            # Gradient clipping
            accumulated_grads, grad_norm = optim.clip_grad_norm(
                accumulated_grads, max_norm=config.grad_clip
            )

            # Update parameters
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state)

            # Track loss
            avg_step_loss = step_loss / config.grad_accum_steps
            total_loss += avg_step_loss

            # Reset accumulation
            accumulated_grads = None
            step_loss = 0.0
            accum_count = 0

            # Logging
            if global_step % config.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / config.log_every
                tok_per_sec = tokens_processed / elapsed

                print(
                    f"Step {global_step:6d}/{config.max_steps} | "
                    f"loss: {avg_loss:.4f} | "
                    f"lr: {lr:.2e} | "
                    f"tok/s: {tok_per_sec:,.0f} | "
                    f"time: {elapsed:.0f}s"
                )
                total_loss = 0.0

            # Generation samples
            if global_step % config.eval_every == 0:
                print("\n--- Generation samples ---")
                prompts = ["def ", "class ", "import ", "for "]
                for prompt in prompts:
                    prompt_ids = tokenizer.encode(prompt)
                    generated_ids = model.generate(
                        prompt_ids, max_new_tokens=60, temperature=0.7
                    )
                    generated = tokenizer.decode(generated_ids)
                    # Clean up for display
                    generated_clean = generated.replace('\n', '\\n')[:120]
                    print(f"  '{prompt}' -> {generated_clean}...")
                print("-" * 70)

            # Save checkpoint
            if global_step % config.save_every == 0:
                ckpt_path = save_dir / f"model_step_{global_step}.safetensors"
                model.save_weights(str(ckpt_path))
                tokenizer.save(str(save_dir / "tokenizer.json"))
                print(f"[Checkpoint saved: {ckpt_path}]")

    # Final save
    final_path = save_dir / "model_final.safetensors"
    model.save_weights(str(final_path))
    tokenizer.save(str(save_dir / "tokenizer.json"))

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"  Total time: {total_time / 3600:.2f} hours")
    print(f"  Total tokens: {tokens_processed:,}")
    print(f"  Average tok/s: {tokens_processed / total_time:,.0f}")
    print(f"  Final model: {final_path}")
    print("=" * 70)

    dataset.close()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Code LLM with streaming data")

    # Data
    parser.add_argument("--data", type=str, default="data/train_tokens_large.bin",
                        help="Path to pre-tokenized binary data")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer JSON")

    # Model
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["tiny", "small", "medium"],
                        help="Model size (tiny=2M, small=57M, medium=350M params)")

    # Training
    parser.add_argument("--steps", type=int, default=20000,
                        help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Micro batch size (per gradient accumulation step)")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length (1024-2048 recommended for code)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup", type=int, default=500,
                        help="Warmup steps")

    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-path", type=str, default="checkpoints")

    args = parser.parse_args()

    config = TrainConfig(
        model_size=args.model_size,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_steps=args.steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_path=args.save_path,
        data_path=args.data,
        tokenizer_path=args.tokenizer,
    )

    train(config)


if __name__ == "__main__":
    main()
