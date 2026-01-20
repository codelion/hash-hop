# Code LLM with Custom Tokenizer

## Goal
Build a small transformer model that can:
1. Learn a custom GUI framework from context
2. Generate code using that framework
3. Make targeted edits to a codebase

## Architecture Overview

### 1. Hybrid Tokenizer

```
Code Input: "def calculate_total(items):"
                    ↓
┌─────────────────────────────────────────┐
│  Symbol Tokenizer (our approach)        │
│  - Function names → single token        │
│  - Variable names → single token        │
│  - Class names → single token           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Syntax Tokenizer (standard)            │
│  - Keywords: def, class, if, for        │
│  - Operators: =, +, -, (, ), :          │
│  - Literals: strings, numbers           │
└─────────────────────────────────────────┘
                    ↓
Output: [DEF] [calculate_total:FUNC] [(] [items:VAR] [)] [:]
```

### 2. Model Architecture

Small transformer with:
- **Embedding dimension**: 256-512
- **Layers**: 6-12
- **Heads**: 4-8
- **Parameters**: ~10M-50M

Key insight: Our tokenizer reduces sequence length dramatically:
- Standard: "calculate_total" → ["calc", "ulate", "_", "total"] (4 tokens)
- Ours: "calculate_total" → [FUNC_42] (1 token)

This means we can fit much more code in context!

### 3. Training Strategy

**Phase 1: Pre-training on code**
- Train on Python/JS codebases
- Objective: Next token prediction
- Learn code syntax and patterns

**Phase 2: Fine-tuning on instruction-to-diff**
- Input: (codebase context, instruction)
- Output: diff/edit to make
- Learn to follow instructions

### 4. Context Structure

```
<|context|>
[Full codebase with custom framework]
</|context|>

<|instruction|>
Create a calculator using the Button and Display components
</|instruction|>

<|response|>
[Generated code using the framework]
</|response|>
```

## Demo 1: Custom GUI Framework

### Setup
Create a simple custom GUI framework:

```python
# gui_framework.py
class Component:
    def __init__(self, id):
        self.id = id
        self.children = []

    def render(self):
        raise NotImplementedError

class Button(Component):
    def __init__(self, id, label, on_click):
        super().__init__(id)
        self.label = label
        self.on_click = on_click

    def render(self):
        return f'<button id="{self.id}" onclick="{self.on_click}">{self.label}</button>'

class Display(Component):
    def __init__(self, id, value=""):
        super().__init__(id)
        self.value = value

    def render(self):
        return f'<div id="{self.id}" class="display">{self.value}</div>'

class App:
    def __init__(self):
        self.components = []
        self.state = {}

    def add(self, component):
        self.components.append(component)

    def render(self):
        return '\n'.join(c.render() for c in self.components)
```

### Task
Given this framework in context, generate:

```python
# calculator.py
from gui_framework import App, Button, Display

app = App()
app.state['display'] = '0'

display = Display('calc-display', app.state['display'])
app.add(display)

for i in range(10):
    btn = Button(f'btn-{i}', str(i), f'press_number({i})')
    app.add(btn)

app.add(Button('btn-plus', '+', 'press_operator("+")'))
app.add(Button('btn-equals', '=', 'calculate()'))

print(app.render())
```

## Demo 2: Code Modification

### Setup
Use a real open-source repo (simplified version)

### Task
Given instruction: "Add a password strength meter to the signup form"

Model should:
1. Find the signup form component
2. Understand the existing patterns
3. Generate a diff that adds the feature

## Implementation Plan

1. **Week 1**: Build hybrid tokenizer
   - Python AST-based symbol extraction
   - Combine with syntax tokens

2. **Week 2**: Train small transformer
   - Use MLX for Apple Silicon
   - Start with 10M params, scale if needed

3. **Week 3**: Fine-tune on instruction data
   - Create synthetic instruction-to-diff pairs
   - Fine-tune on target codebases

4. **Week 4**: Build demos
   - Custom GUI framework example
   - Real repo modification example

## Memory/Compute Estimates

For 10M parameter model:
- Model weights: ~40MB (fp32) / ~20MB (fp16)
- Training batch: ~1GB for batch_size=8, seq_len=2048
- Total RAM needed: ~4-8GB

Should be feasible on Mac with 36GB RAM!
