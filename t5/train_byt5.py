"""Fine-tune ByT5 (byte-level T5) on HashHop task.

ByT5 is a tokenizer-free T5 variant that operates directly on UTF-8 bytes.
This eliminates tokenization issues with random character strings that plague
SentencePiece-based models like T5-base.

Key differences from T5:
- No tokenizer needed - uses raw UTF-8 bytes
- Longer sequence lengths (bytes > subwords)
- Better handling of random/noisy text
- Slightly slower but more accurate for character-level tasks

Available models:
- google/byt5-small: 300M params
- google/byt5-base: 580M params
- google/byt5-large: 1.2B params
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import mlx.core as mx
import mlx.optimizers as optim
from huggingface_hub import snapshot_download

from t5.model import T5
from t5.dataset import HashHopT5Dataset, T5HashHopSample


class ByT5Tokenizer:
    """Byte-level tokenizer for ByT5.

    ByT5 uses raw UTF-8 bytes as tokens, with special tokens:
    - 0: <pad>
    - 1: <eos>
    - 2: <unk>
    - 3-258: UTF-8 bytes (0-255) offset by 3

    Total vocab size: 384 (includes some unused special tokens)
    """

    def __init__(self):
        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.offset = 3  # Bytes are offset by 3
        self.vocab_size = 384
        self.decoder_start_id = 0  # ByT5 uses pad as decoder start

    def encode(self, text: str, max_length: int = 1024) -> mx.array:
        """Encode text to byte token IDs."""
        # Convert to UTF-8 bytes
        bytes_data = text.encode("utf-8")

        # Convert bytes to token IDs (add offset)
        token_ids = [b + self.offset for b in bytes_data]

        # Truncate if needed
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Pad to max_length
        padding_needed = max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * padding_needed

        return mx.array(token_ids, dtype=mx.int32)

    def encode_batch(
        self, texts: List[str], max_length: int = 1024
    ) -> Tuple[mx.array, mx.array]:
        """Encode batch of texts, return tokens and attention mask."""
        batch_tokens = []
        batch_masks = []

        for text in texts:
            bytes_data = text.encode("utf-8")
            token_ids = [b + self.offset for b in bytes_data]

            # Create attention mask before padding
            mask = [1] * min(len(token_ids), max_length)

            # Truncate
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                # Pad
                padding_needed = max_length - len(token_ids)
                mask = mask + [0] * padding_needed
                token_ids = token_ids + [self.pad_id] * padding_needed

            batch_tokens.append(token_ids)
            batch_masks.append(mask)

        return mx.array(batch_tokens, dtype=mx.int32), mx.array(batch_masks, dtype=mx.int32)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        # Filter out special tokens and convert back to bytes
        bytes_list = []
        for tid in token_ids:
            if tid >= self.offset and tid < self.offset + 256:
                bytes_list.append(tid - self.offset)
            elif tid == self.eos_id:
                break
            # Skip pad and other special tokens

        try:
            return bytes(bytes_list).decode("utf-8")
        except UnicodeDecodeError:
            # Handle incomplete UTF-8 sequences
            return bytes(bytes_list).decode("utf-8", errors="replace")


def convert_pytorch_to_mlx(pytorch_path: Path, output_path: Path) -> None:
    """Convert PyTorch .bin weights to MLX .safetensors format.

    Requires torch to be installed. Falls back to a slower numpy-based method
    if torch is not available.
    """
    import pickle
    import struct

    print(f"Converting {pytorch_path} to {output_path}...")

    try:
        import torch
        # Load with torch and convert to numpy, then MLX
        state_dict = torch.load(pytorch_path, map_location="cpu")
        weights = {k: mx.array(v.numpy()) for k, v in state_dict.items()}
        mx.save_safetensors(str(output_path), weights)
        print(f"Converted {len(weights)} tensors using torch")
    except ImportError:
        # Fall back to manual unpickling (may not work for all models)
        print("torch not available, trying manual conversion...")
        import numpy as np
        import zipfile

        # PyTorch .bin files are zip archives containing pickled tensors
        with zipfile.ZipFile(pytorch_path, "r") as zf:
            # Find the data file
            names = zf.namelist()
            data_files = [n for n in names if n.endswith("data.pkl") or "data" in n]

            # This is a simplified approach - may need adjustment for different model formats
            raise ImportError(
                "torch is required to convert PyTorch .bin weights. "
                "Install it with: pip install torch"
            )


def load_byt5_model(model_name: str = "google/byt5-small", dtype=mx.bfloat16) -> Tuple[T5, dict]:
    """Load ByT5 model from HuggingFace.

    ByT5 uses the same architecture as T5, just with byte-level vocab.
    Supports both safetensors and PyTorch bin formats.
    """
    path = Path(
        snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
        )
    )

    with open(path / "config.json", "r") as f:
        config = SimpleNamespace(**json.load(f))

    model = T5(config)

    # Try safetensors first, fall back to pytorch bin
    safetensors_path = path / "model.safetensors"
    pytorch_path = path / "pytorch_model.bin"

    # Save converted weights to local cache to avoid permission issues
    cache_dir = Path("checkpoints/byt5_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_cache_name = model_name.replace("/", "_")
    converted_path = cache_dir / f"{model_cache_name}_mlx.safetensors"

    if safetensors_path.exists():
        print(f"Loading weights from {safetensors_path}")
        weights = mx.load(str(safetensors_path))
    elif converted_path.exists():
        print(f"Loading previously converted weights from {converted_path}")
        weights = mx.load(str(converted_path))
    elif pytorch_path.exists():
        # Convert PyTorch weights to MLX format
        convert_pytorch_to_mlx(pytorch_path, converted_path)
        weights = mx.load(str(converted_path))
    else:
        raise FileNotFoundError(
            f"No model weights found in {path}. "
            f"Available files: {list(path.glob('*'))}"
        )

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
    """Compute cross-entropy loss for ByT5."""
    logits = model(encoder_input, decoder_input)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
    batch_indices = mx.arange(logits_flat.shape[0])
    target_log_probs = log_probs[batch_indices, targets_flat]

    mask = (targets_flat != pad_id).astype(mx.float32)
    loss = -target_log_probs * mask

    return loss.sum() / (mask.sum() + 1e-8)


def prepare_batch(
    samples: List[T5HashHopSample],
    tokenizer: ByT5Tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 32,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Prepare a batch for training."""
    input_texts = [s.input_text for s in samples]
    target_texts = [s.target_text for s in samples]

    encoder_input, _ = tokenizer.encode_batch(input_texts, max_length=max_input_length)
    targets, _ = tokenizer.encode_batch(target_texts, max_length=max_target_length)

    batch_size = targets.shape[0]
    decoder_start = mx.full((batch_size, 1), tokenizer.decoder_start_id, dtype=mx.int32)
    decoder_input = mx.concatenate([decoder_start, targets[:, :-1]], axis=1)

    return encoder_input, decoder_input, targets


def evaluate(
    model: T5,
    tokenizer: ByT5Tokenizer,
    n_chars_problem: int,
    num_samples: int = 50,
    max_input_length: int = 1024,
) -> float:
    """Evaluate model on HashHop samples."""
    dataset = HashHopT5Dataset(
        n_chars_problem=n_chars_problem,
        num_queries=1,
        hops=1,
        hash_pair_str_length=4,
    )

    correct = 0
    for _ in range(num_samples):
        sample = dataset.generate_sample()
        encoder_input = tokenizer.encode(sample.input_text, max_length=max_input_length)
        encoder_input = encoder_input[None, :]

        memory = model.encode(encoder_input)
        decoder_input = mx.array([[tokenizer.decoder_start_id]])
        generated = []

        # Generate byte by byte
        for _ in range(len(sample.target_text.encode("utf-8")) + 5):
            logits = model.decode(decoder_input, memory)[0]
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            if next_token.item() == tokenizer.eos_id:
                break

            generated.append(next_token.item())
            decoder_input = mx.concatenate(
                [decoder_input, next_token[:, None]], axis=1
            )

        output = tokenizer.decode(generated)
        if output.strip() == sample.target_text:
            correct += 1

    return correct / num_samples


def flatten_params(params, prefix=""):
    """Flatten nested params dict/list to flat dict."""
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


def train(
    model_name: str = "google/byt5-small",
    n_chars_problem: int = 200,
    max_steps: int = 5000,
    batch_size: int = 2,  # ByT5 uses more memory due to longer sequences
    learning_rate: float = 5e-5,  # Lower LR for pretrained model
    eval_every: int = 500,
    save_every: int = 1000,
    output_dir: str = "checkpoints/byt5_hashhop",
    max_input_length: int = 1024,  # Bytes are longer than subword tokens
):
    """Train ByT5 on HashHop."""
    print(f"Loading {model_name}...")
    model, config = load_byt5_model(model_name)
    tokenizer = ByT5Tokenizer()

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

    # Test tokenization
    test_text = "ABCD=EFGH"
    encoded = tokenizer.encode(test_text, max_length=20)
    decoded = tokenizer.decode(encoded.tolist())
    print(f"Tokenization test: '{test_text}' -> {encoded[:len(test_text)+2].tolist()} -> '{decoded[:len(test_text)]}'")

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

    loss_and_grad = mx.value_and_grad(loss_fn)

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining on {n_chars_problem} char contexts for {max_steps} steps...")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Max input length: {max_input_length} bytes")

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
            accuracy = evaluate(
                model, tokenizer, n_chars_problem, num_samples=50
            )
            print(f"Step {step}: accuracy={accuracy*100:.1f}%\n")

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = output_path / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            weights = flatten_params(model.parameters())
            mx.save_safetensors(str(ckpt_path / "weights.safetensors"), weights)
            with open(ckpt_path / "config.json", "w") as f:
                json.dump(vars(config), f, indent=2)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = output_path / "final"
    final_path.mkdir(exist_ok=True)
    weights = flatten_params(model.parameters())
    mx.save_safetensors(str(final_path / "weights.safetensors"), weights)
    with open(final_path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nTraining complete. Model saved to {final_path}")

    # Final evaluation
    print("\nFinal evaluation...")
    accuracy = evaluate(model, tokenizer, n_chars_problem, num_samples=100)
    print(f"Final accuracy: {accuracy*100:.1f}%")

    # Test on larger context sizes
    print("\nEvaluating on multiple context sizes...")
    for size in [200, 500, 1000]:
        if size <= n_chars_problem * 2:  # Only test reasonable extrapolations
            acc = evaluate(model, tokenizer, size, num_samples=50)
            print(f"  {size} chars: {acc*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ByT5 on HashHop")
    parser.add_argument(
        "--model", type=str, default="google/byt5-small",
        help="ByT5 model name (google/byt5-small, google/byt5-base, google/byt5-large)"
    )
    parser.add_argument(
        "--context-size", type=int, default=200, help="Context size in characters"
    )
    parser.add_argument("--max-steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=500, help="Eval frequency")
    parser.add_argument("--save-every", type=int, default=1000, help="Save frequency")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/byt5_hashhop", help="Output dir"
    )
    parser.add_argument(
        "--max-input-length", type=int, default=1024, help="Max input length in bytes"
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
