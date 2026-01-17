"""Evaluate T5 on HashHop."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from t5.model import T5
from t5.dataset import HashHopT5Dataset


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
        self.decoder_start_id = self._tokenizer.pad_token_id

    def encode(self, text: str, max_length: int = 512) -> mx.array:
        tokens = self._tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return mx.array(tokens["input_ids"][0])

    def decode(self, token_ids) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)


def load_model(checkpoint_path: str, model_name: str = "t5-base", dtype=mx.bfloat16):
    """Load model from checkpoint or pretrained."""
    # First load base config
    path = Path(
        snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
    )

    with open(path / "config.json", "r") as f:
        config = SimpleNamespace(**json.load(f))

    model = T5(config)

    # Check if checkpoint has custom weights
    ckpt_path = Path(checkpoint_path)
    if (ckpt_path / "weights.safetensors").exists():
        print(f"Loading weights from {ckpt_path}")
        flat_weights = mx.load(str(ckpt_path / "weights.safetensors"))

        # Unflatten the weights - handle nested dict/list structure
        def unflatten_params(flat_weights):
            """Convert flat key format to nested dict/list structure."""
            result = {}
            for key, value in flat_weights.items():
                parts = key.split(".")
                current = result

                for i, part in enumerate(parts[:-1]):
                    next_part = parts[i + 1]

                    # Check if we need a list (next part is numeric)
                    if next_part.isdigit():
                        if part not in current:
                            current[part] = []
                        # Ensure list is long enough
                        idx = int(next_part)
                        while len(current[part]) <= idx:
                            current[part].append({})
                        current = current[part]
                    elif part.isdigit():
                        # Current part is an index into a list
                        idx = int(part)
                        current = current[idx]
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]

                # Set the final value
                final_key = parts[-1]
                current[final_key] = value.astype(dtype)

            return result

        weights = unflatten_params(flat_weights)
        model.update(weights)
        print(f"Loaded {len(flat_weights)} weight tensors")
    else:
        # Load pretrained weights
        weights = mx.load(str(path / "model.safetensors"))
        weights = T5.sanitize(weights)
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        model.load_weights(list(weights.items()))

    return model


def evaluate(
    model: T5,
    tokenizer: T5Tokenizer,
    n_chars_problem: int = 1000,
    num_samples: int = 100,
    max_input_length: int = 512,
    verbose: bool = False,
):
    """Evaluate model on HashHop."""
    dataset = HashHopT5Dataset(
        n_chars_problem=n_chars_problem,
        num_queries=1,
        hops=1,
        hash_pair_str_length=4,
    )

    correct = 0
    partial_matches = 0

    for i in range(num_samples):
        sample = dataset.generate_sample()

        # Encode input
        encoder_input = tokenizer.encode(sample.input_text, max_length=max_input_length)
        encoder_input = encoder_input[None, :]

        # Encode the context
        memory = model.encode(encoder_input)

        # Generate autoregressively
        decoder_input = mx.array([[tokenizer.decoder_start_id]])
        generated = []

        for _ in range(len(sample.target_text) + 5):
            logits = model.decode(decoder_input, memory)[0]
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            if next_token.item() == tokenizer.eos_id:
                break

            generated.append(next_token.item())
            decoder_input = mx.concatenate(
                [decoder_input, next_token[:, None]], axis=1
            )

        # Decode and compare
        output = tokenizer.decode(generated).strip()
        expected = sample.target_text

        if output == expected:
            correct += 1
            if verbose:
                print(f"[{i+1}] CORRECT: '{output}' == '{expected}'")
        else:
            # Check partial match
            match_len = 0
            for a, b in zip(output, expected):
                if a == b:
                    match_len += 1
                else:
                    break
            if match_len > 0:
                partial_matches += 1
            if verbose:
                print(f"[{i+1}] WRONG: got '{output}' expected '{expected}'")

    accuracy = correct / num_samples
    partial_rate = partial_matches / num_samples

    print(f"\nResults on {n_chars_problem} char context:")
    print(f"  Exact match: {correct}/{num_samples} = {accuracy*100:.1f}%")
    print(f"  Partial match: {partial_matches}/{num_samples} = {partial_rate*100:.1f}%")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 on HashHop")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--model", type=str, default="t5-base", help="Base model name")
    parser.add_argument(
        "--context-size", type=int, default=1000, help="Context size in characters"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max-input-length", type=int, default=512, help="Max input token length"
    )
    parser.add_argument("--verbose", action="store_true", help="Print each sample")

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model)
    tokenizer = T5Tokenizer(args.model)

    print(f"Evaluating on {args.num_samples} samples...")
    evaluate(
        model,
        tokenizer,
        n_chars_problem=args.context_size,
        num_samples=args.num_samples,
        max_input_length=args.max_input_length,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
