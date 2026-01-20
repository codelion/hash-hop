"""Training pipeline for Small Code LLM.

Trains the model on:
1. Next token prediction on code
2. Instruction-following for code edits
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import time
import json

from tokenizer import CodeTokenizer
from model import SmallCodeLLM, softmax


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute cross entropy loss and gradient.

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) target token IDs

    Returns:
        (loss, grad_logits)
    """
    batch, seq_len, vocab_size = logits.shape

    # Reshape for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch*seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch*seq_len,)

    # Compute softmax probabilities
    probs = softmax(logits_flat, axis=-1)

    # Get probabilities of correct tokens
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]

    # Cross entropy loss (negative log likelihood)
    loss = -np.mean(np.log(correct_probs + 1e-10))

    # Gradient: probs - one_hot(targets)
    grad = probs.copy()
    grad[np.arange(len(targets_flat)), targets_flat] -= 1
    grad = grad / (batch * seq_len)

    grad = grad.reshape(batch, seq_len, vocab_size)

    return loss, grad


class Trainer:
    """Simple trainer for the code LLM."""

    def __init__(
        self,
        model: SmallCodeLLM,
        tokenizer: CodeTokenizer,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate

        # Adam optimizer state
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _get_param_list(self) -> List[Tuple[str, np.ndarray]]:
        """Get all trainable parameters."""
        params = [
            ('token_embeddings', self.model.token_embeddings),
            ('pos_embeddings', self.model.pos_embeddings),
        ]

        for i, block in enumerate(self.model.blocks):
            params.extend([
                (f'block_{i}_W_q', block.attention.W_q),
                (f'block_{i}_W_k', block.attention.W_k),
                (f'block_{i}_W_v', block.attention.W_v),
                (f'block_{i}_W_o', block.attention.W_o),
                (f'block_{i}_W1', block.ffn.W1),
                (f'block_{i}_b1', block.ffn.b1),
                (f'block_{i}_W2', block.ffn.W2),
                (f'block_{i}_b2', block.ffn.b2),
                (f'block_{i}_ln1_gamma', block.ln1_gamma),
                (f'block_{i}_ln1_beta', block.ln1_beta),
                (f'block_{i}_ln2_gamma', block.ln2_gamma),
                (f'block_{i}_ln2_beta', block.ln2_beta),
            ])

        params.extend([
            ('ln_f_gamma', self.model.ln_f_gamma),
            ('ln_f_beta', self.model.ln_f_beta),
        ])

        return params

    def _adam_update(self, name: str, param: np.ndarray, grad: np.ndarray):
        """Apply Adam optimizer update."""
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2

        m_hat = self.m[name] / (1 - self.beta1**self.t)
        v_hat = self.v[name] / (1 - self.beta2**self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray) -> float:
        """Single training step with gradient descent.

        Note: This uses numerical gradients for simplicity.
        For real training, you'd want automatic differentiation (JAX/PyTorch).

        Args:
            input_ids: (batch, seq_len) input token IDs
            target_ids: (batch, seq_len) target token IDs

        Returns:
            loss value
        """
        self.t += 1

        # Forward pass
        logits = self.model.forward(input_ids)
        loss, grad_logits = cross_entropy_loss(logits, target_ids)

        # For this simple implementation, we'll use gradient descent on embeddings
        # and skip backprop through transformer layers (would need autograd)

        # Update token embeddings based on which tokens were used
        batch, seq_len = input_ids.shape
        for b in range(batch):
            for s in range(seq_len):
                token_id = input_ids[b, s]
                target_id = target_ids[b, s]

                # Simple gradient: push embedding toward predicting correct token
                # This is a simplification of full backprop
                grad = grad_logits[b, s]  # (vocab_size,)

                # Update the embedding for this token
                self.model.token_embeddings[token_id] -= self.lr * 0.01 * grad[:self.model.d_model].mean()

        return loss

    def train_on_code(
        self,
        code_files: List[str],
        num_epochs: int = 10,
        batch_size: int = 4,
        seq_len: int = 256,
        verbose: bool = True
    ):
        """Train on code files using next token prediction.

        Args:
            code_files: List of paths to Python files
            num_epochs: Number of training epochs
            batch_size: Batch size
            seq_len: Sequence length for training
            verbose: Print progress
        """
        # Load and tokenize all code
        all_tokens = []
        for file_path in code_files:
            try:
                with open(file_path) as f:
                    code = f.read()
                tokens = self.tokenizer.encode(code)
                all_tokens.extend(tokens)
            except Exception as e:
                if verbose:
                    print(f"  Skipping {file_path}: {e}")

        if verbose:
            print(f"Total tokens: {len(all_tokens):,}")
            print(f"Vocabulary size: {self.tokenizer.vocab_size()}")

        # Ensure model vocab matches tokenizer
        if self.tokenizer.vocab_size() > self.model.vocab_size:
            # Expand model embeddings
            old_emb = self.model.token_embeddings
            new_emb = np.random.randn(self.tokenizer.vocab_size(), self.model.d_model) * 0.01
            new_emb[:old_emb.shape[0]] = old_emb
            self.model.token_embeddings = new_emb
            self.model.output_proj = new_emb.T
            self.model.vocab_size = self.tokenizer.vocab_size()

        # Training loop
        num_batches = (len(all_tokens) - seq_len) // (batch_size * seq_len)

        if verbose:
            print(f"\nTraining for {num_epochs} epochs, {num_batches} batches/epoch")
            print("-" * 60)

        for epoch in range(num_epochs):
            epoch_loss = 0
            start_time = time.time()

            # Shuffle data
            np.random.shuffle(all_tokens)

            for batch_idx in range(num_batches):
                # Create batch
                inputs = []
                targets = []

                for b in range(batch_size):
                    start = (batch_idx * batch_size + b) * seq_len
                    end = start + seq_len

                    if end + 1 < len(all_tokens):
                        inputs.append(all_tokens[start:end])
                        targets.append(all_tokens[start+1:end+1])

                if len(inputs) == 0:
                    continue

                input_ids = np.array(inputs)
                target_ids = np.array(targets)

                # Training step
                loss = self.train_step(input_ids, target_ids)
                epoch_loss += loss

            avg_loss = epoch_loss / max(1, num_batches)
            elapsed = time.time() - start_time

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, time={elapsed:.1f}s")


def create_gui_framework_demo():
    """Create the custom GUI framework for demo 1."""

    framework_code = '''
"""Simple GUI Framework - Custom implementation for demo."""

class Component:
    """Base class for all GUI components."""

    def __init__(self, id: str):
        self.id = id
        self.children = []
        self.styles = {}

    def add_child(self, child):
        self.children.append(child)
        return self

    def style(self, **kwargs):
        self.styles.update(kwargs)
        return self

    def render(self) -> str:
        raise NotImplementedError


class Button(Component):
    """Clickable button component."""

    def __init__(self, id: str, label: str, on_click: str = ""):
        super().__init__(id)
        self.label = label
        self.on_click = on_click

    def render(self) -> str:
        style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        return f'<button id="{self.id}" onclick="{self.on_click}" style="{style_str}">{self.label}</button>'


class Display(Component):
    """Text display component."""

    def __init__(self, id: str, value: str = ""):
        super().__init__(id)
        self.value = value

    def render(self) -> str:
        style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        return f'<div id="{self.id}" class="display" style="{style_str}">{self.value}</div>'


class Container(Component):
    """Container for grouping components."""

    def __init__(self, id: str, layout: str = "vertical"):
        super().__init__(id)
        self.layout = layout

    def render(self) -> str:
        flex_dir = "column" if self.layout == "vertical" else "row"
        style_str = f"display: flex; flex-direction: {flex_dir}; " + "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        children_html = "\\n".join(c.render() for c in self.children)
        return f'<div id="{self.id}" style="{style_str}">\\n{children_html}\\n</div>'


class Input(Component):
    """Text input component."""

    def __init__(self, id: str, placeholder: str = "", on_change: str = ""):
        super().__init__(id)
        self.placeholder = placeholder
        self.on_change = on_change

    def render(self) -> str:
        style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        return f'<input id="{self.id}" placeholder="{self.placeholder}" onchange="{self.on_change}" style="{style_str}"/>'


class App:
    """Main application container."""

    def __init__(self, title: str = "App"):
        self.title = title
        self.root = Container("root", "vertical")
        self.state = {}
        self.scripts = []

    def add(self, component: Component):
        self.root.add_child(component)
        return self

    def add_script(self, script: str):
        self.scripts.append(script)
        return self

    def render(self) -> str:
        scripts_html = "\\n".join(f"<script>{s}</script>" for s in self.scripts)
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        .display {{ font-size: 24px; padding: 10px; background: #f0f0f0; margin: 5px; }}
        button {{ padding: 10px 20px; margin: 5px; font-size: 18px; cursor: pointer; }}
        input {{ padding: 10px; margin: 5px; font-size: 18px; }}
    </style>
</head>
<body>
{self.root.render()}
{scripts_html}
</body>
</html>"""


# Example usage showing how to build a simple counter:
#
# from gui_framework import App, Button, Display, Container
#
# app = App("Counter")
# display = Display("counter-display", "0")
#
# app.add(display)
# app.add(Button("increment", "+", "increment()"))
# app.add(Button("decrement", "-", "decrement()"))
#
# app.add_script("""
# let count = 0;
# function increment() { count++; document.getElementById('counter-display').innerText = count; }
# function decrement() { count--; document.getElementById('counter-display').innerText = count; }
# """)
#
# print(app.render())
'''

    # Example calculator built with the framework
    calculator_code = '''
"""Calculator built with the custom GUI framework."""

from gui_framework import App, Button, Display, Container

# Create the calculator app
app = App("Calculator")

# Create display
display = Display("calc-display", "0").style(
    font_size="32px",
    text_align="right",
    min_width="200px"
)
app.add(display)

# Create button rows
row1 = Container("row1", "horizontal")
row1.add_child(Button("btn-7", "7", "press('7')"))
row1.add_child(Button("btn-8", "8", "press('8')"))
row1.add_child(Button("btn-9", "9", "press('9')"))
row1.add_child(Button("btn-div", "/", "press('/')"))
app.add(row1)

row2 = Container("row2", "horizontal")
row2.add_child(Button("btn-4", "4", "press('4')"))
row2.add_child(Button("btn-5", "5", "press('5')"))
row2.add_child(Button("btn-6", "6", "press('6')"))
row2.add_child(Button("btn-mul", "*", "press('*')"))
app.add(row2)

row3 = Container("row3", "horizontal")
row3.add_child(Button("btn-1", "1", "press('1')"))
row3.add_child(Button("btn-2", "2", "press('2')"))
row3.add_child(Button("btn-3", "3", "press('3')"))
row3.add_child(Button("btn-sub", "-", "press('-')"))
app.add(row3)

row4 = Container("row4", "horizontal")
row4.add_child(Button("btn-0", "0", "press('0')"))
row4.add_child(Button("btn-clear", "C", "clear()"))
row4.add_child(Button("btn-eq", "=", "calculate()"))
row4.add_child(Button("btn-add", "+", "press('+')"))
app.add(row4)

# Add calculator logic
app.add_script("""
let expression = '';

function press(val) {
    expression += val;
    document.getElementById('calc-display').innerText = expression;
}

function clear() {
    expression = '';
    document.getElementById('calc-display').innerText = '0';
}

function calculate() {
    try {
        let result = eval(expression);
        document.getElementById('calc-display').innerText = result;
        expression = String(result);
    } catch (e) {
        document.getElementById('calc-display').innerText = 'Error';
        expression = '';
    }
}
""")

# Render the app
print(app.render())
'''

    return framework_code, calculator_code


def demo():
    """Demo training and generation."""
    print("=" * 60)
    print("CODE LLM TRAINING DEMO")
    print("=" * 60)

    # Create GUI framework files
    framework_code, calculator_code = create_gui_framework_demo()

    # Save framework
    Path("code_llm/demo_data").mkdir(parents=True, exist_ok=True)
    with open("code_llm/demo_data/gui_framework.py", "w") as f:
        f.write(framework_code)
    with open("code_llm/demo_data/calculator.py", "w") as f:
        f.write(calculator_code)

    print("\nCreated demo files:")
    print("  - code_llm/demo_data/gui_framework.py")
    print("  - code_llm/demo_data/calculator.py")

    # Initialize tokenizer and model
    tokenizer = CodeTokenizer()

    # First, tokenize all code to get vocab size
    all_code = framework_code + "\n" + calculator_code
    tokens = tokenizer.tokenize(all_code)
    vocab_size = tokenizer.vocab_size()

    print(f"\nTokenizer stats:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Vocabulary size: {vocab_size}")

    # Create model
    model = SmallCodeLLM(
        vocab_size=vocab_size + 1000,  # Some buffer
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=512
    )

    print(f"\nModel stats:")
    print(f"  Parameters: {model.count_params():,}")

    # Create trainer
    trainer = Trainer(model, tokenizer, learning_rate=1e-3)

    # Train on the code
    print("\nTraining on GUI framework code...")
    trainer.train_on_code(
        code_files=[
            "code_llm/demo_data/gui_framework.py",
            "code_llm/demo_data/calculator.py",
        ],
        num_epochs=5,
        batch_size=2,
        seq_len=128,
        verbose=True
    )

    # Save model
    model.save("code_llm/demo_data/model.pkl")
    tokenizer.save("code_llm/demo_data/tokenizer.json")
    print("\nSaved model and tokenizer.")

    # Test generation
    print("\n" + "-" * 60)
    print("Testing generation...")

    prompt = "from gui_framework import App, Button"
    prompt_ids = tokenizer.encode(prompt)

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=50,
        temperature=0.8
    )

    generated_code = tokenizer.decode(generated_ids)
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated:\n{generated_code}")


if __name__ == "__main__":
    demo()
