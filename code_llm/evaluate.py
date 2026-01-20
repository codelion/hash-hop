"""Evaluation and demo script for the trained Python Code LLM.

Demonstrates the model's ability to:
1. Complete Python code
2. Generate functions from signatures
3. Continue class definitions
"""

import mlx.core as mx
from pathlib import Path
from mlx_model import SmallCodeLLM, create_model
from tokenizer import CodeTokenizer


def load_model(
    model_path: str = "checkpoints/model_final.safetensors",
    tokenizer_path: str = "checkpoints/tokenizer.json",
) -> tuple[SmallCodeLLM, CodeTokenizer]:
    """Load trained model and tokenizer."""
    # Load tokenizer
    tokenizer = CodeTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size()}")

    # Create model with same architecture
    model = create_model(tokenizer.vocab_size(), "small")
    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")
    print(f"  Parameters: {model.count_params():,}")

    return model, tokenizer


def generate_completion(
    model: SmallCodeLLM,
    tokenizer: CodeTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    """Generate code completion for a prompt."""
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated_ids)


def demo_completions():
    """Demonstrate various code completions."""
    print("=" * 70)
    print("Python Code LLM - Demonstration")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model()

    # Test prompts
    prompts = [
        # Function definitions
        "def calculate_sum(",
        "def fibonacci(n):",
        "def read_file(filename):",

        # Class definitions
        "class Calculator:",
        "class DataProcessor:",

        # Control flow
        "for i in range(",
        "if __name__ == ",
        "try:",

        # Imports and structure
        "import os\nimport sys\n\ndef main():",
        "from typing import List, Dict\n\nclass ",
    ]

    print("\n" + "-" * 70)
    print("Code Completion Examples")
    print("-" * 70)

    for prompt in prompts:
        print(f"\n>>> Prompt: {repr(prompt)}")
        print("-" * 40)

        completion = generate_completion(
            model, tokenizer, prompt,
            max_tokens=60,
            temperature=0.7,
        )
        print(completion)
        print()


def interactive_demo():
    """Interactive code completion demo."""
    print("=" * 70)
    print("Interactive Python Code Completion")
    print("=" * 70)
    print("Type your code prompt and press Enter.")
    print("Type 'quit' to exit.\n")

    model, tokenizer = load_model()

    while True:
        try:
            prompt = input("\n>>> ")
            if prompt.lower() == 'quit':
                break

            completion = generate_completion(
                model, tokenizer, prompt,
                max_tokens=80,
                temperature=0.7,
            )
            print("\nGenerated:")
            print("-" * 40)
            print(completion)

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nGoodbye!")


def evaluate_on_samples():
    """Evaluate model on specific completion tasks."""
    print("=" * 70)
    print("Evaluation: Code Completion Quality")
    print("=" * 70)

    model, tokenizer = load_model()

    # Test cases with expected patterns
    test_cases = [
        {
            "prompt": "def add(a, b):\n    ",
            "description": "Simple function body",
            "expected_contains": ["return", "+"],
        },
        {
            "prompt": "class Person:\n    def __init__(self",
            "description": "Constructor signature",
            "expected_contains": ["self", ")", ":"],
        },
        {
            "prompt": "for i in range(10):\n    ",
            "description": "Loop body",
            "expected_contains": ["i", "print"],
        },
        {
            "prompt": "if x > 0:\n    return ",
            "description": "Conditional return",
            "expected_contains": ["x", "return"],
        },
    ]

    results = []
    for test in test_cases:
        completion = generate_completion(
            model, tokenizer, test["prompt"],
            max_tokens=40,
            temperature=0.5,  # Lower temp for more deterministic output
        )

        # Check if expected patterns are present
        matches = sum(1 for pattern in test["expected_contains"]
                     if pattern in completion)
        score = matches / len(test["expected_contains"])
        results.append(score)

        print(f"\n{test['description']}")
        print(f"Prompt: {repr(test['prompt'])}")
        print(f"Completion: {completion[:100]}...")
        print(f"Pattern match: {score:.0%}")

    avg_score = sum(results) / len(results)
    print(f"\n{'=' * 70}")
    print(f"Average pattern match score: {avg_score:.0%}")


def compare_with_without_context():
    """Compare generation with and without code context.

    This demonstrates the advantage of treating identifiers as single tokens:
    the model should better track variable names across context.
    """
    print("=" * 70)
    print("Context Awareness Demo")
    print("=" * 70)

    model, tokenizer = load_model()

    # Test: Does the model use the variable name from context?
    context_tests = [
        {
            "context": "user_data = load_user_data()\n",
            "prompt": "print(",
            "description": "Should reference 'user_data' variable",
        },
        {
            "context": "def process_items(item_list):\n    result = []\n    for item in item_list:\n",
            "prompt": "        result.append(",
            "description": "Should reference 'item' from loop",
        },
        {
            "context": "class DataProcessor:\n    def __init__(self, config):\n        self.config = config\n\n    def process(self):\n",
            "prompt": "        return self.",
            "description": "Should reference 'config' attribute",
        },
    ]

    for test in context_tests:
        full_prompt = test["context"] + test["prompt"]

        print(f"\n{test['description']}")
        print(f"Context:\n{test['context']}")
        print(f"Prompt: {repr(test['prompt'])}")
        print("-" * 40)

        completion = generate_completion(
            model, tokenizer, full_prompt,
            max_tokens=30,
            temperature=0.5,
        )
        # Show just the completion part
        completion_only = completion[len(test["context"]):]
        print(f"Completion: {completion_only}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Python Code LLM")
    parser.add_argument("--demo", action="store_true", help="Run demo completions")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--context", action="store_true", help="Context awareness demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    args = parser.parse_args()

    if args.all or (not any([args.demo, args.interactive, args.evaluate, args.context])):
        demo_completions()
        print("\n")
        evaluate_on_samples()
        print("\n")
        compare_with_without_context()
    else:
        if args.demo:
            demo_completions()
        if args.interactive:
            interactive_demo()
        if args.evaluate:
            evaluate_on_samples()
        if args.context:
            compare_with_without_context()
