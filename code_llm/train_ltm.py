"""Training script for LTM (Long-Term Memory) Model.

Trains the model end-to-end on a code retrieval + generation task:
1. Given: memory bank of code snippets + query
2. Model must: attend to relevant snippets and generate correct output

This is similar to how MagicLabs LTM-2 learns to use massive context.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from pathlib import Path
import time
import numpy as np
from typing import List, Tuple, Dict
import random
import re

from ltm_model import LTMModel
from tokenizer import CodeTokenizer


def create_code_qa_dataset(tokenizer: CodeTokenizer, num_samples: int = 1000):
    """Create a synthetic code QA dataset for training.

    Each sample has:
    - memory: List of code snippets (some relevant, some distractors)
    - query: A question/prompt about the code
    - target: The expected answer/completion

    This teaches the model to retrieve relevant snippets and generate answers.
    """
    # Simple code patterns the model should learn
    code_templates = [
        # Function definitions
        ("def add(a, b):\n    return a + b", "add", "addition"),
        ("def subtract(a, b):\n    return a - b", "subtract", "subtraction"),
        ("def multiply(a, b):\n    return a * b", "multiply", "multiplication"),
        ("def divide(a, b):\n    return a / b", "divide", "division"),
        ("def square(x):\n    return x * x", "square", "square"),
        ("def cube(x):\n    return x * x * x", "cube", "cube"),
        ("def double(x):\n    return x * 2", "double", "double"),
        ("def negate(x):\n    return -x", "negate", "negate"),
        ("def absolute(x):\n    return abs(x)", "absolute", "absolute value"),
        ("def increment(x):\n    return x + 1", "increment", "increment"),

        # String functions
        ("def upper(s):\n    return s.upper()", "upper", "uppercase"),
        ("def lower(s):\n    return s.lower()", "lower", "lowercase"),
        ("def reverse(s):\n    return s[::-1]", "reverse", "reverse"),
        ("def length(s):\n    return len(s)", "length", "length"),

        # List functions
        ("def first(lst):\n    return lst[0]", "first", "first element"),
        ("def last(lst):\n    return lst[-1]", "last", "last element"),
        ("def sum_list(lst):\n    return sum(lst)", "sum_list", "sum"),
        ("def max_list(lst):\n    return max(lst)", "max_list", "maximum"),
        ("def min_list(lst):\n    return min(lst)", "min_list", "minimum"),
        ("def count(lst):\n    return len(lst)", "count", "count"),
    ]

    # Distractor code (irrelevant snippets)
    distractors = [
        "import os\nimport sys",
        "class Config:\n    pass",
        "# This is a comment",
        "x = 42\ny = 'hello'",
        "for i in range(10):\n    pass",
        "if True:\n    pass",
        "while False:\n    break",
        "try:\n    pass\nexcept:\n    pass",
    ]

    samples = []
    for _ in range(num_samples):
        # Pick a target code template
        code, func_name, description = random.choice(code_templates)

        # Create memory: target + distractors
        memory_codes = [code]
        # Add some random distractors
        num_distractors = random.randint(3, 8)
        for _ in range(num_distractors):
            # Either a distractor or another code template
            if random.random() < 0.5:
                memory_codes.append(random.choice(distractors))
            else:
                other = random.choice(code_templates)
                if other[1] != func_name:  # Don't add the same function
                    memory_codes.append(other[0])

        # Shuffle memory
        random.shuffle(memory_codes)

        # Create query and target
        query_templates = [
            f"# Call the {func_name} function\n",
            f"# Use {description}\nresult = ",
            f"# {func_name}\nx = ",
        ]
        query = random.choice(query_templates)
        target = f"{func_name}("

        # Tokenize everything
        memory_tokens = [tokenizer.encode(c) for c in memory_codes]
        query_tokens = tokenizer.encode(query)
        target_tokens = tokenizer.encode(target)

        samples.append({
            "memory": memory_tokens,
            "query": query_tokens,
            "target": target_tokens,
            "func_name": func_name,
        })

    return samples


def collate_batch(
    samples: List[Dict],
    max_query_len: int = 64,
    max_target_len: int = 32,
    pad_id: int = 0,
) -> Tuple[mx.array, List[List[mx.array]], mx.array]:
    """Collate samples into a batch.

    Returns:
        input_ids: (batch, seq_len) - query + target concatenated
        memories: List of memory token lists per sample
        targets: (batch, seq_len) - shifted targets for loss
    """
    batch_inputs = []
    batch_targets = []
    batch_memories = []

    for sample in samples:
        query = sample["query"]
        target = sample["target"]

        # Truncate/pad query
        if len(query) > max_query_len:
            query = query[:max_query_len]
        else:
            query = query + [pad_id] * (max_query_len - len(query))

        # Truncate/pad target
        if len(target) > max_target_len:
            target = target[:max_target_len]
        else:
            target = target + [pad_id] * (max_target_len - len(target))

        # Input is query + target[:-1], target is query[1:] + target
        input_seq = query + target[:-1]
        target_seq = query[1:] + target

        batch_inputs.append(input_seq)
        batch_targets.append(target_seq)

        # Convert memory to mx arrays
        memory_arrays = [mx.array(m) for m in sample["memory"]]
        batch_memories.append(memory_arrays)

    return (
        mx.array(batch_inputs),
        batch_memories,
        mx.array(batch_targets),
    )


def loss_fn(
    model: LTMModel,
    input_ids: mx.array,
    memory: mx.array,
    targets: mx.array,
    pad_id: int = 0,
) -> mx.array:
    """Compute cross-entropy loss."""
    logits, _ = model(input_ids, memory)

    # Flatten
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Create mask for non-padding tokens
    mask = (targets_flat != pad_id).astype(mx.float32)

    # Cross entropy
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)

    return loss


def train(
    num_steps: int = 1000,
    batch_size: int = 8,
    lr: float = 1e-3,
    d_model: int = 128,
    n_layers: int = 2,
    log_every: int = 50,
    eval_every: int = 200,
):
    """Train the LTM model."""
    print("=" * 60)
    print("LTM Model Training")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = CodeTokenizer()
    # Add some basic symbols
    for sym in ["add", "subtract", "multiply", "divide", "square", "cube",
                "double", "negate", "absolute", "increment", "upper", "lower",
                "reverse", "length", "first", "last", "sum_list", "max_list",
                "min_list", "count", "result", "x", "a", "b", "s", "lst"]:
        tokenizer._add_symbol(sym)

    print(f"\nTokenizer vocab size: {tokenizer.vocab_size()}")

    # Create dataset
    print("Creating training dataset...")
    train_data = create_code_qa_dataset(tokenizer, num_samples=2000)
    print(f"  Training samples: {len(train_data)}")

    # Create model
    model = LTMModel(
        vocab_size=tokenizer.vocab_size(),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        d_ff=d_model * 4,
        max_seq_len=128,
    )
    print(f"\nModel parameters: {model.count_params():,}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    # Loss and grad function
    def compute_loss(model, input_ids, memory, targets):
        return loss_fn(model, input_ids, memory, targets)

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 60)

    start_time = time.time()
    total_loss = 0.0
    step = 0

    while step < num_steps:
        # Sample batch
        batch_samples = random.sample(train_data, min(batch_size, len(train_data)))
        input_ids, batch_memories, targets = collate_batch(batch_samples)

        # Encode memories for each sample and stack
        # For simplicity, use the first sample's memory for the whole batch
        # (In a real implementation, we'd handle variable-length memories)
        memory = model.encode_memory(batch_memories[0])

        # Forward and backward
        loss, grads = loss_and_grad(model, input_ids, memory, targets)

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        step += 1

        # Logging
        if step % log_every == 0:
            avg_loss = total_loss / log_every
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | loss: {avg_loss:.4f} | time: {elapsed:.0f}s")
            total_loss = 0.0

        # Evaluation
        if step % eval_every == 0:
            print("\n--- Evaluation ---")
            eval_model(model, tokenizer, train_data[:5])
            print("-" * 60)

    print(f"\nTraining complete!")
    return model, tokenizer


def eval_model(model: LTMModel, tokenizer: CodeTokenizer, samples: List[Dict]):
    """Evaluate model on samples."""
    for i, sample in enumerate(samples[:3]):
        # Encode memory
        memory = model.encode_memory([mx.array(m) for m in sample["memory"]])

        # Generate from query
        query_tokens = sample["query"]
        generated = model.generate(memory, query_tokens, max_new_tokens=20, temperature=0.5)

        # Decode
        query_str = tokenizer.decode(query_tokens)
        generated_str = tokenizer.decode(generated[len(query_tokens):])
        expected = sample["func_name"]

        print(f"\n  Sample {i+1}:")
        print(f"    Query: {query_str[:50]}...")
        print(f"    Expected: {expected}(")
        print(f"    Generated: {generated_str[:30]}...")

        # Check if correct function was retrieved
        if expected in generated_str:
            print(f"    ✓ Correct!")
        else:
            print(f"    ✗ Wrong")


def demo():
    """Quick demo of the trained model."""
    print("\n" + "=" * 60)
    print("LTM Model Demo")
    print("=" * 60)

    # Train a small model
    model, tokenizer = train(
        num_steps=500,
        batch_size=4,
        lr=1e-3,
        d_model=128,
        n_layers=2,
        log_every=50,
        eval_every=100,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Create some test cases
    test_memory = [
        "def add(a, b):\n    return a + b",
        "def multiply(a, b):\n    return a * b",
        "class Calculator:\n    pass",
        "import math",
        "x = 42",
    ]
    memory_tokens = [mx.array(tokenizer.encode(c)) for c in test_memory]
    memory = model.encode_memory(memory_tokens)

    queries = [
        "# Call the add function\nresult = ",
        "# Use multiplication\nx = ",
    ]

    for query in queries:
        query_tokens = tokenizer.encode(query)
        generated = model.generate(memory, query_tokens, max_new_tokens=15, temperature=0.5)
        generated_str = tokenizer.decode(generated[len(query_tokens):])
        print(f"\nQuery: {query}")
        print(f"Generated: {generated_str}")


if __name__ == "__main__":
    demo()
