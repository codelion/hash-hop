"""Training script for Python Code LLM using MLX.

Trains on Python code with proper autograd and Adam optimizer.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
import time
import json
import math
from typing import List, Iterator, Tuple
from dataclasses import dataclass

from mlx_model import SmallCodeLLM, create_model
from tokenizer import CodeTokenizer


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_size: str = "small"  # tiny, small, medium
    max_seq_len: int = 512

    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_clip: float = 1.0

    # Data
    data_path: str = "data/python_code"

    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    save_path: str = "checkpoints"


def get_batch(
    tokens: List[int],
    batch_size: int,
    seq_len: int,
) -> Tuple[mx.array, mx.array]:
    """Get a random batch of sequences.

    Returns:
        inputs: (batch, seq_len) - input tokens
        targets: (batch, seq_len) - target tokens (shifted by 1)
    """
    # Random starting positions
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        # Data too short, repeat it
        tokens = tokens * (seq_len * 2 // len(tokens) + 1)
        max_start = len(tokens) - seq_len - 1

    starts = [mx.random.randint(0, max_start).item() for _ in range(batch_size)]

    inputs = []
    targets = []
    for start in starts:
        inputs.append(tokens[start:start + seq_len])
        targets.append(tokens[start + 1:start + seq_len + 1])

    return mx.array(inputs), mx.array(targets)


def loss_fn(model: SmallCodeLLM, inputs: mx.array, targets: mx.array) -> mx.array:
    """Compute cross-entropy loss."""
    logits = model(inputs)
    # Reshape for cross entropy
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Cross entropy loss
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss


def train(config: TrainConfig, code_files: List[str]):
    """Train the model on Python code files."""
    print("=" * 60)
    print("MLX Code LLM Training")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = CodeTokenizer()

    # Load and tokenize all code
    print("\nLoading and tokenizing code...")
    all_tokens = []
    for path in code_files:
        try:
            with open(path) as f:
                code = f.read()
            tokens = tokenizer.encode(code)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.token_to_id.get('<NEWLINE>', 6))  # Separator
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size()}")

    # Create model
    model = create_model(tokenizer.vocab_size(), config.model_size)
    print(f"\nModel: {config.model_size}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_layers: {model.n_layers}")

    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Loss and grad function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training state
    state = [model.state, optimizer.state]

    def train_step(tokens: List[int]) -> float:
        """Single training step."""
        inputs, targets = get_batch(tokens, config.batch_size, config.max_seq_len)

        # Forward and backward
        loss, grads = loss_and_grad(model, inputs, targets)

        # Gradient clipping
        grads, _ = optim.clip_grad_norm(grads, max_norm=config.grad_clip)

        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        return loss.item()

    # Training loop
    print(f"\nTraining for {config.max_steps} steps...")
    print("-" * 60)

    Path(config.save_path).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    total_loss = 0
    tokens_per_sec = 0

    for step in range(1, config.max_steps + 1):
        # Learning rate warmup
        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
            optimizer.learning_rate = lr

        # Training step
        step_start = time.time()
        loss = train_step(all_tokens)
        step_time = time.time() - step_start

        total_loss += loss
        tokens_processed = config.batch_size * config.max_seq_len
        tokens_per_sec = tokens_processed / step_time

        # Logging
        if step % config.log_every == 0:
            avg_loss = total_loss / config.log_every
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | loss: {avg_loss:.4f} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"lr: {optimizer.learning_rate:.2e} | "
                  f"time: {elapsed:.0f}s")
            total_loss = 0

        # Evaluation / generation sample
        if step % config.eval_every == 0:
            print("\n--- Generation sample ---")
            prompt = "def "
            prompt_ids = tokenizer.encode(prompt)
            generated_ids = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8)
            generated = tokenizer.decode(generated_ids)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated[:200]}...")
            print("-" * 60)

        # Save checkpoint
        if step % config.save_every == 0:
            save_path = Path(config.save_path) / f"model_step_{step}.safetensors"
            model.save_weights(str(save_path))
            tokenizer.save(str(Path(config.save_path) / "tokenizer.json"))
            print(f"Saved checkpoint to {save_path}")

    # Final save
    final_path = Path(config.save_path) / "model_final.safetensors"
    model.save_weights(str(final_path))
    tokenizer.save(str(Path(config.save_path) / "tokenizer.json"))
    print(f"\nTraining complete! Final model saved to {final_path}")

    return model, tokenizer


def load_python_files(directory: str, max_files: int = 1000) -> List[str]:
    """Load Python files from a directory."""
    files = []
    for path in Path(directory).rglob("*.py"):
        files.append(str(path))
        if len(files) >= max_files:
            break
    return files


def demo_training():
    """Demo training on local Python files."""
    print("Demo: Training on local Python files")
    print("=" * 60)

    # Find Python files in the current project
    project_root = Path(__file__).parent.parent
    py_files = load_python_files(str(project_root), max_files=50)
    print(f"Found {len(py_files)} Python files")

    if len(py_files) < 5:
        print("Not enough Python files found. Creating synthetic data...")
        # Create some synthetic Python code for training
        synthetic_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True

def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total

def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, x):
        self.result += x
        return self

    def subtract(self, x):
        self.result -= x
        return self

    def multiply(self, x):
        self.result *= x
        return self

    def get_result(self):
        return self.result

def process_list(items):
    """Process a list of items."""
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def main():
    calc = Calculator()
    calc.add(10).subtract(3).multiply(2)
    print(f"Result: {calc.get_result()}")

    numbers = [1, 2, 3, 4, 5]
    total = calculate_sum(numbers)
    print(f"Sum: {total}")

if __name__ == "__main__":
    main()
'''
        # Write synthetic data
        data_dir = Path("code_llm/demo_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            with open(data_dir / f"synthetic_{i}.py", "w") as f:
                f.write(synthetic_code)
        py_files = load_python_files(str(data_dir))

    # Use smaller config for demo
    config = TrainConfig(
        model_size="tiny",
        max_seq_len=256,
        batch_size=2,
        learning_rate=1e-3,
        max_steps=200,
        log_every=10,
        eval_every=50,
        save_every=100,
        save_path="code_llm/checkpoints",
    )

    # Train
    model, tokenizer = train(config, py_files)

    # Test generation
    print("\n" + "=" * 60)
    print("Testing trained model")
    print("=" * 60)

    prompts = [
        "def ",
        "class ",
        "for ",
        "if ",
    ]

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.7)
        generated = tokenizer.decode(generated_ids)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")


def train_from_pretokenized(
    data_path: str = "data/train_tokens.bin",
    tokenizer_path: str = "data/tokenizer.json",
    config: TrainConfig = None,
):
    """Train from pre-tokenized data file."""
    import struct

    print("=" * 60)
    print("Training from pre-tokenized data")
    print("=" * 60)

    # Load tokenizer
    tokenizer = CodeTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size()}")

    # Load tokens
    print(f"Loading tokens from {data_path}...")
    with open(data_path, "rb") as f:
        num_tokens = struct.unpack("Q", f.read(8))[0]
        tokens = []
        for _ in range(num_tokens):
            token = struct.unpack("I", f.read(4))[0]
            tokens.append(token)
    print(f"Loaded {len(tokens):,} tokens")

    if config is None:
        config = TrainConfig(
            model_size="small",
            max_seq_len=512,
            batch_size=8,
            learning_rate=3e-4,
            warmup_steps=200,
            max_steps=2000,
            log_every=20,
            eval_every=200,
            save_every=500,
            save_path="code_llm/checkpoints",
        )

    # Create model
    model = create_model(tokenizer.vocab_size(), config.model_size)
    print(f"\nModel: {config.model_size}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_layers: {model.n_layers}")

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Loss and grad function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(tokens: List[int]) -> float:
        inputs, targets = get_batch(tokens, config.batch_size, config.max_seq_len)
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, _ = optim.clip_grad_norm(grads, max_norm=config.grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss.item()

    # Training loop
    print(f"\nTraining for {config.max_steps} steps...")
    print("-" * 60)

    Path(config.save_path).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    total_loss = 0

    for step in range(1, config.max_steps + 1):
        # Learning rate warmup
        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
            optimizer.learning_rate = lr

        # Training step
        step_start = time.time()
        loss = train_step(tokens)
        step_time = time.time() - step_start

        total_loss += loss
        tokens_per_sec = config.batch_size * config.max_seq_len / step_time

        # Logging
        if step % config.log_every == 0:
            avg_loss = total_loss / config.log_every
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | loss: {avg_loss:.4f} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"lr: {optimizer.learning_rate:.2e} | "
                  f"time: {elapsed:.0f}s")
            total_loss = 0

        # Generation sample
        if step % config.eval_every == 0:
            print("\n--- Generation samples ---")
            for prompt in ["def ", "class ", "for item in "]:
                prompt_ids = tokenizer.encode(prompt)
                generated_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
                generated = tokenizer.decode(generated_ids)
                print(f"'{prompt}' -> {generated[:100]}...")
            print("-" * 60)

        # Save checkpoint
        if step % config.save_every == 0:
            save_path = Path(config.save_path) / f"model_step_{step}.safetensors"
            model.save_weights(str(save_path))
            tokenizer.save(str(Path(config.save_path) / "tokenizer.json"))
            print(f"Saved checkpoint to {save_path}")

    # Final save
    final_path = Path(config.save_path) / "model_final.safetensors"
    model.save_weights(str(final_path))
    tokenizer.save(str(Path(config.save_path) / "tokenizer.json"))
    print(f"\nTraining complete! Final model saved to {final_path}")

    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_tokens.bin")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--demo", action="store_true", help="Run demo training on local files")
    args = parser.parse_args()

    if args.demo:
        demo_training()
    else:
        config = TrainConfig(
            model_size=args.model_size,
            max_seq_len=args.seq_len,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.steps,
        )
        train_from_pretokenized(args.data, args.tokenizer, config)
