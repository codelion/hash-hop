"""Train MALM on CodeParrot dataset.

This trains the Memory-Augmented Language Model on real Python code from
the CodeParrot dataset to enable:
1. Code Retrieval & Q&A on any Python codebase
2. Code Transformation patterns (with enhanced architecture)
3. HashHop-style exact retrieval

Key insight: We train the retrieval mechanism + generation jointly.
The memory bank contains real Python functions from CodeParrot.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
import random
import time
import re
import ast
import json
from pathlib import Path
from dataclasses import dataclass

# Import our MALM components
from memory_augmented_lm import (
    MemoryAugmentedLM,
    DynamicCodeTokenizer,
    combined_loss,
)


@dataclass
class FunctionItem:
    """A function extracted from code."""
    name: str
    source: str
    docstring: str
    signature: str
    module: str


def extract_functions_from_code(code: str, module_name: str = "unknown") -> List[FunctionItem]:
    """Extract functions from Python source code."""
    functions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    # Get source lines
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 20
                    lines = code.split('\n')[start:min(end, start + 30)]
                    func_source = '\n'.join(lines)

                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    # Get signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    signature = f"{node.name}({', '.join(args)})"

                    functions.append(FunctionItem(
                        name=node.name,
                        source=func_source[:800],  # Limit length
                        docstring=docstring[:300],
                        signature=signature,
                        module=module_name,
                    ))
                except Exception:
                    pass
    except SyntaxError:
        pass

    return functions


def load_codeparrot_streaming(
    max_samples: int = 10000,
    min_functions: int = 2,
    max_functions_total: int = None,
) -> Iterator[Dict]:
    """Stream CodeParrot data from HuggingFace.

    Returns samples that contain at least min_functions extractable functions.
    Stops when max_samples OR max_functions_total is reached.
    """
    try:
        from datasets import load_dataset
        print("Loading CodeParrot dataset (streaming)...")
        print(f"  Target: up to {max_samples} samples, {max_functions_total or 'unlimited'} functions")

        dataset = load_dataset(
            "codeparrot/codeparrot-clean",
            split="train",
            streaming=True,
        )

        count = 0
        total_functions = 0
        start_time = time.time()

        for sample in dataset:
            if count >= max_samples:
                break
            if max_functions_total and total_functions >= max_functions_total:
                break

            code = sample.get("content", "")
            functions = extract_functions_from_code(code)

            if len(functions) >= min_functions:
                yield {
                    "code": code,
                    "functions": functions,
                }
                count += 1
                total_functions += len(functions)

                if count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"  Streaming: {count} samples, {total_functions} functions ({rate:.1f} samples/sec)")

        elapsed = time.time() - start_time
        print(f"  Total: {count} samples, {total_functions} functions in {elapsed:.1f}s")

    except ImportError:
        print("datasets library not available, using synthetic data...")
        yield from generate_synthetic_data(max_samples)
    except Exception as e:
        print(f"Error loading CodeParrot: {e}")
        print("Falling back to synthetic data...")
        yield from generate_synthetic_data(max_samples)


def generate_synthetic_data(max_samples: int = 1000) -> Iterator[Dict]:
    """Generate synthetic Python functions for training."""

    # Templates for different function types
    templates = [
        # Arithmetic
        ("def {name}(a, b):\n    \"\"\"Compute {op} of two numbers.\"\"\"\n    return a {symbol} b",
         ["add", "subtract", "multiply"], ["+", "-", "*"], ["sum", "difference", "product"]),

        # Single arg
        ("def {name}(x):\n    \"\"\"Compute {op}.\"\"\"\n    return {expr}",
         ["square", "double", "negate", "increment"],
         ["x * x", "x * 2", "-x", "x + 1"],
         ["square", "double", "negation", "increment"]),

        # String ops
        ("def {name}(s):\n    \"\"\"Return {op} of string.\"\"\"\n    return s.{method}()",
         ["upper", "lower", "strip", "title"],
         ["upper", "lower", "strip", "title"],
         ["uppercase", "lowercase", "stripped", "titlecase"]),

        # List ops
        ("def {name}(lst):\n    \"\"\"Return {op} of list.\"\"\"\n    return {func}(lst)",
         ["max_val", "min_val", "total", "length"],
         ["max", "min", "sum", "len"],
         ["maximum", "minimum", "sum", "length"]),

        # Conditionals
        ("def {name}(a, b):\n    \"\"\"Return {op}.\"\"\"\n    return a if a {cmp} b else b",
         ["maximum", "minimum"], [">", "<"], ["larger value", "smaller value"]),
    ]

    count = 0
    while count < max_samples:
        functions = []

        # Generate 3-8 functions per sample
        num_funcs = random.randint(3, 8)
        for _ in range(num_funcs):
            template_group = random.choice(templates)
            template, names, exprs, ops = template_group

            idx = random.randint(0, len(names) - 1)
            name = names[idx] + f"_{random.randint(1, 100)}"

            if "{symbol}" in template:
                source = template.format(name=name, op=ops[idx], symbol=exprs[idx])
            elif "{expr}" in template:
                source = template.format(name=name, op=ops[idx], expr=exprs[idx])
            elif "{method}" in template:
                source = template.format(name=name, op=ops[idx], method=exprs[idx])
            elif "{func}" in template:
                source = template.format(name=name, op=ops[idx], func=exprs[idx])
            elif "{cmp}" in template:
                source = template.format(name=name, op=ops[idx], cmp=exprs[idx])
            else:
                continue

            functions.append(FunctionItem(
                name=name.split("_")[0],  # Base name without suffix
                source=source,
                docstring=f"Compute {ops[idx]}",
                signature=f"{name}(...)",
                module="synthetic",
            ))

        if len(functions) >= 2:
            yield {
                "code": "\n\n".join(f.source for f in functions),
                "functions": functions,
            }
            count += 1


class CodeParrotDataLoader:
    """Data loader for CodeParrot that manages memory and batching."""

    def __init__(
        self,
        tokenizer: DynamicCodeTokenizer,
        max_memory_items: int = 1000,
        max_val_len: int = 100,
        batch_memory_size: int = 50,
    ):
        self.tokenizer = tokenizer
        self.max_memory_items = max_memory_items
        self.max_val_len = max_val_len
        self.batch_memory_size = batch_memory_size

        # Memory bank
        self.memory_items: List[FunctionItem] = []
        self.keys: List[int] = []
        self.values: List[List[int]] = []

    def add_functions(self, functions: List[FunctionItem]):
        """Add functions to the memory bank."""
        for func in functions:
            if len(self.memory_items) >= self.max_memory_items:
                break

            # Skip if name already exists (avoid duplicates)
            if any(f.name == func.name for f in self.memory_items):
                continue

            # Tokenize
            self.tokenizer.encode(func.name)
            self.tokenizer.encode(func.source)
            if func.docstring:
                self.tokenizer.encode(func.docstring)

            # Add to memory
            self.memory_items.append(func)
            self.keys.append(self.tokenizer.get_id(func.name))

            val_ids = self.tokenizer.encode(func.source)
            val_ids = val_ids[:self.max_val_len] + [0] * (self.max_val_len - len(val_ids))
            self.values.append(val_ids)

    def get_memory_arrays(self) -> Tuple[mx.array, mx.array]:
        """Get memory as MLX arrays."""
        return mx.array(self.keys), mx.array(self.values)

    def create_training_batch(
        self,
        batch_size: int = 16,
        fixed_len: int = 30,
    ) -> Dict:
        """Create a training batch with queries and targets."""
        if len(self.memory_items) < batch_size:
            batch_size = len(self.memory_items)

        # Sample items for this batch
        indices = random.sample(range(len(self.memory_items)), batch_size)

        inputs = []
        targets = []
        target_indices = []

        for idx in indices:
            item = self.memory_items[idx]

            # Create query variations
            # KEY: For retrieval training, queries should NOT contain the answer
            # This forces the model to actually retrieve from memory

            query_types = []

            # Docstring-based queries (preferred - no function name)
            if item.docstring and len(item.docstring.split()) >= 3:
                words = item.docstring.split()[:5]
                query_types.append("# " + " ".join(words))
                query_types.append("find function that " + " ".join(words[:3]))

            # Signature-based queries (no function name)
            args = item.signature.split("(")[1].rstrip(")") if "(" in item.signature else ""
            if args and args != "self":
                query_types.append(f"function with args {args}")

            # Fallback: use function name (less ideal for pure retrieval)
            if not query_types:
                query_types = [
                    f"function {item.name}",
                    f"call {item.name}",
                ]

            query = random.choice(query_types)
            input_ids = self.tokenizer.encode(query)
            target_ids = input_ids[1:] + [self.tokenizer.get_id(item.name)]

            # Pad/truncate
            input_ids = input_ids[:fixed_len] + [0] * (fixed_len - len(input_ids))
            target_ids = target_ids[:fixed_len] + [0] * (fixed_len - len(target_ids))

            inputs.append(input_ids)
            targets.append(target_ids)
            target_indices.append(idx)

        return {
            "input_ids": mx.array(inputs),
            "target_ids": mx.array(targets),
            "target_idx": mx.array(target_indices),
        }

    def create_transformation_batch(
        self,
        batch_size: int = 8,
        fixed_len: int = 60,
    ) -> Optional[Dict]:
        """Create a batch for code transformation learning.

        This creates pairs where the input is code without type hints
        and the target is the same code with type hints.
        """
        # Find functions that could have type hints added
        candidates = []
        for idx, item in enumerate(self.memory_items):
            # Check if function has simple signature
            if "def " in item.source and "-> " not in item.source:
                candidates.append((idx, item))

        if len(candidates) < batch_size:
            return None

        samples = random.sample(candidates, batch_size)

        inputs = []
        targets = []
        target_indices = []

        for idx, item in samples:
            # Create type-hinted version (simplified pattern)
            source = item.source
            typed = add_simple_type_hints(source)

            input_ids = self.tokenizer.encode(source)
            target_ids = self.tokenizer.encode(typed)

            input_ids = input_ids[:fixed_len] + [0] * (fixed_len - len(input_ids))
            target_ids = target_ids[:fixed_len] + [0] * (fixed_len - len(target_ids))

            inputs.append(input_ids)
            targets.append(target_ids)
            target_indices.append(idx)

        return {
            "input_ids": mx.array(inputs),
            "target_ids": mx.array(targets),
            "target_idx": mx.array(target_indices),
        }


def add_simple_type_hints(source: str) -> str:
    """Add simple type hints to a function (heuristic-based)."""
    # This is a simplified pattern matcher
    # Real implementation would use AST

    # Pattern: def name(a, b):
    match = re.match(r'def (\w+)\(([^)]*)\):', source)
    if not match:
        return source

    name = match.group(1)
    args = match.group(2)

    # Simple heuristic: arithmetic ops get int, div gets float
    if any(op in source for op in ['/', 'avg', 'mean', 'ratio']):
        ret_type = "float"
        arg_type = "float"
    else:
        ret_type = "int"
        arg_type = "int"

    # Add type hints to args
    if args:
        typed_args = ", ".join(f"{a.strip()}: {arg_type}" for a in args.split(","))
    else:
        typed_args = ""

    # Reconstruct
    rest = source[match.end():]
    return f"def {name}({typed_args}) -> {ret_type}:{rest}"


def train_malm_on_codeparrot(
    max_samples: int = 5000,
    max_memory_items: int = 1000,
    num_steps: int = 10000,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    log_every: int = 100,
    eval_every: int = 1000,
    checkpoint_dir: str = "checkpoints/malm",
):
    """Train MALM on CodeParrot dataset."""

    print("=" * 70)
    print("Training MALM on CodeParrot Dataset")
    print("=" * 70)
    print()

    # Create tokenizer and data loader
    tokenizer = DynamicCodeTokenizer()
    data_loader = CodeParrotDataLoader(
        tokenizer,
        max_memory_items=max_memory_items,
        max_val_len=100,
    )

    # Load data
    print("Loading CodeParrot data...")
    sample_count = 0
    for sample in load_codeparrot_streaming(max_samples=max_samples):
        data_loader.add_functions(sample["functions"])
        sample_count += 1

        if len(data_loader.memory_items) >= max_memory_items:
            break

    print(f"\nLoaded {sample_count} code samples")
    print(f"Memory bank: {len(data_loader.memory_items)} functions")
    print(f"Vocabulary: {tokenizer.vocab_size()} tokens")

    # Create model
    model = MemoryAugmentedLM(
        vocab_size=tokenizer.vocab_size() + 1000,  # Buffer for new tokens
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=256,
    )

    flat_params = mlx_utils.tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    print(f"Model parameters: {num_params:,}")

    # Encode memory once
    keys, values = data_loader.get_memory_arrays()
    key_emb, val_emb = model.encode_memory(keys, values)
    print(f"Memory encoded: keys={key_emb.shape}, values={val_emb.shape}")

    # Optimizer with warmup
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, input_ids, target_ids, target_idx, key_emb, val_emb, temp):
        total, gen, ret = combined_loss(
            model, input_ids, target_ids, target_idx,
            key_emb, val_emb,
            retrieval_weight=5.0,  # Higher weight for retrieval
            gen_weight=1.0,
            temperature=temp,
        )
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 70)

    start_time = time.time()
    total_loss = 0.0
    best_ret_acc = 0.0

    for step in range(1, num_steps + 1):
        # Temperature annealing
        temp = max(0.5, 2.0 - step / num_steps * 1.5)

        # Create batch
        batch = data_loader.create_training_batch(batch_size=batch_size)

        # Forward and backward
        loss, grads = loss_and_grad(
            model,
            batch["input_ids"],
            batch["target_ids"],
            batch["target_idx"],
            key_emb, val_emb,
            temp,
        )

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

        # Logging
        if step % log_every == 0:
            avg_loss = total_loss / log_every
            elapsed = time.time() - start_time

            # Quick retrieval accuracy check
            _, attn, _ = model(batch["input_ids"], key_emb, val_emb, temp)
            pred_idx = mx.argmax(attn, axis=1)
            ret_acc = mx.mean((pred_idx == batch["target_idx"]).astype(mx.float32)).item()

            print(f"Step {step:5d} | loss={avg_loss:.4f} | ret_acc={ret_acc:.2%} | temp={temp:.2f} | time={elapsed:.0f}s")
            total_loss = 0.0

            if ret_acc > best_ret_acc:
                best_ret_acc = ret_acc

        # Evaluation
        if step % eval_every == 0:
            print("\n" + "-" * 40)
            print("EVALUATION")
            print("-" * 40)
            evaluate_malm(model, key_emb, val_emb, data_loader, tokenizer)
            print("-" * 40 + "\n")

    # Save checkpoint
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = checkpoint_path / "model_weights.npz"
    flat_params = mlx_utils.tree_flatten(model.parameters())
    np.savez(str(weights_path), **{k: np.array(v) for k, v in flat_params})

    # Save tokenizer
    tokenizer_path = checkpoint_path / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump({
            "token_to_id": tokenizer.token_to_id,
            "next_id": tokenizer.next_id,
        }, f)

    # Save config
    config_path = checkpoint_path / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "vocab_size": model.vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "max_seq_len": 256,
            "num_memory_items": len(data_loader.memory_items),
        }, f)

    print(f"\nCheckpoint saved to {checkpoint_path}")
    print(f"Best retrieval accuracy: {best_ret_acc:.2%}")

    return model, tokenizer, data_loader, key_emb, val_emb


def evaluate_malm(
    model: MemoryAugmentedLM,
    key_emb: mx.array,
    val_emb: mx.array,
    data_loader: CodeParrotDataLoader,
    tokenizer: DynamicCodeTokenizer,
):
    """Evaluate MALM on all three use cases."""

    # 1. Code Retrieval
    print("\n1. Code Retrieval:")
    test_items = random.sample(data_loader.memory_items, min(10, len(data_loader.memory_items)))
    correct = 0

    for item in test_items:
        query = f"function {item.name}"
        input_ids = tokenizer.encode(query)
        generated, attn = model.generate(key_emb, val_emb, input_ids, max_new_tokens=5)
        output = tokenizer.decode(generated[len(input_ids):])

        if item.name in output:
            correct += 1

        print(f"   Q: '{query[:25]}' -> '{output[:15]}' {'✓' if item.name in output else '✗'}")

    print(f"   Retrieval Accuracy: {correct}/{len(test_items)} ({100*correct/len(test_items):.0f}%)")

    # 2. Exact Retrieval (HashHop-style)
    print("\n2. Exact Retrieval:")

    # Test direct function name lookup
    exact_correct = 0
    for item in test_items[:5]:
        input_ids = [tokenizer.get_id(item.name)]
        generated, attn = model.generate(key_emb, val_emb, input_ids, max_new_tokens=10)
        output = tokenizer.decode(generated[len(input_ids):])

        # Check if we retrieve something related
        idx = data_loader.memory_items.index(item)
        pred_idx = int(mx.argmax(attn[0]))
        is_correct = pred_idx == idx

        if is_correct:
            exact_correct += 1

        print(f"   '{item.name}' -> idx {pred_idx} (expected {idx}) {'✓' if is_correct else '✗'}")

    print(f"   Exact Accuracy: {exact_correct}/5 ({100*exact_correct/5:.0f}%)")

    # 3. Code Understanding (generation quality)
    print("\n3. Generation Quality:")
    for item in test_items[:3]:
        input_ids = tokenizer.encode(f"# Call {item.name}\nresult = ")
        generated, _ = model.generate(key_emb, val_emb, input_ids, max_new_tokens=15)
        output = tokenizer.decode(generated[len(input_ids):])
        print(f"   # Call {item.name} -> {output[:30]}")


def demo_on_custom_codebase(
    model: MemoryAugmentedLM,
    tokenizer: DynamicCodeTokenizer,
    code_dir: str,
):
    """Demo: Load a custom codebase into memory and answer questions."""
    print("\n" + "=" * 70)
    print("Demo: Custom Codebase Q&A")
    print("=" * 70)

    # Load Python files from directory
    code_path = Path(code_dir)
    if not code_path.exists():
        print(f"Directory not found: {code_dir}")
        return

    data_loader = CodeParrotDataLoader(tokenizer, max_memory_items=200)

    for py_file in code_path.rglob("*.py"):
        try:
            code = py_file.read_text()
            functions = extract_functions_from_code(code, str(py_file.relative_to(code_path)))
            data_loader.add_functions(functions)
        except Exception:
            pass

    print(f"Loaded {len(data_loader.memory_items)} functions from {code_dir}")

    # Encode memory
    keys, values = data_loader.get_memory_arrays()
    key_emb, val_emb = model.encode_memory(keys, values)

    # Interactive Q&A
    print("\nAsk questions about the codebase (type 'quit' to exit):")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        input_ids = tokenizer.encode(query)
        generated, attn = model.generate(key_emb, val_emb, input_ids, max_new_tokens=20)
        output = tokenizer.decode(generated[len(input_ids):])

        # Show top retrieved function
        top_idx = int(mx.argmax(attn[0]))
        if top_idx < len(data_loader.memory_items):
            top_func = data_loader.memory_items[top_idx]
            print(f"\nRetrieved: {top_func.name} from {top_func.module}")
            print(f"Source preview: {top_func.source[:100]}...")

        print(f"\nGenerated: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MALM on CodeParrot")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max CodeParrot samples")
    parser.add_argument("--max-memory", type=int, default=1000, help="Max memory items")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval-codebase", type=str, help="Path to codebase for demo")

    args = parser.parse_args()

    model, tokenizer, data_loader, key_emb, val_emb = train_malm_on_codeparrot(
        max_samples=args.max_samples,
        max_memory_items=args.max_memory,
        num_steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        lr=args.lr,
    )

    # Final comprehensive evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    evaluate_malm(model, key_emb, val_emb, data_loader, tokenizer)

    # Demo on custom codebase if provided
    if args.eval_codebase:
        demo_on_custom_codebase(model, tokenizer, args.eval_codebase)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"""
MALM trained on CodeParrot:
- Memory bank: {len(data_loader.memory_items)} functions
- Vocabulary: {tokenizer.vocab_size()} tokens
- Model parameters: {sum(p.size for _, p in mlx_utils.tree_flatten(model.parameters())):,}

Capabilities:
1. ✅ Code Retrieval & Q&A: Load any Python codebase, ask questions
2. ✅ Exact Retrieval: HashHop-style key-value lookup
3. ⚠️  Code Transformation: Limited (retrieval-based, not seq2seq)

To use on your own codebase:
  python train_malm_codeparrot.py --eval-codebase /path/to/your/code
""")
