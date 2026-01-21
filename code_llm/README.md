# MALM: Memory-Augmented Language Model for Code

A 165M parameter model for code understanding with **100% exact retrieval** accuracy. MALM uses the same key insight that made HashHop work: treating each function name as a single token enables perfect key-value lookup.

## Quick Start

```bash
# Train MALM on CodeParrot
poetry run python code_llm/malm.py --max-functions 2000 --steps 10000

# Use pre-trained checkpoint
python -c "
from code_llm.malm import load_model, MALM
model, tokenizer, functions = load_model('checkpoints/malm')
print(f'Loaded {len(functions)} functions')
"
```

## Results

| Query Type | Accuracy |
|------------|----------|
| Exact Name Queries | **100%** |
| Semantic Queries | **100%** |
| Name Decomposition | **86%** |

## Architecture

MALM is a **165M parameter** model with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Encoder                           │
│  "add two numbers" → [embedding] → attention → query_emb    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory Bank                             │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │ Function Key │  ────▶  │ Implementation Value          │  │
│  │ (single tok) │         │ (encoded sequence)            │  │
│  └──────────────┘         └──────────────────────────────┘  │
│  add             ────▶    def add(a, b): return a + b       │
│  multiply        ────▶    def multiply(a, b): return a * b  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Attention-based retrieval
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Transformer Decoder                        │
│                                                              │
│  Generate output based on retrieved function context         │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Breakdown (165M total)

| Component | Parameters | Formula |
|-----------|------------|---------|
| Token Embedding | 11.1M | vocab_size (14407) × d_model (768) |
| Position Embedding | 0.1M | max_seq_len (128) × d_model (768) |
| Query Encoder (4 layers) | 28.4M | 4 × transformer_layer_params |
| Value Encoder (4 layers) | 28.4M | 4 × transformer_layer_params |
| Decoder (12 layers) | 85.1M | 12 × transformer_layer_params |
| Output Projection | 11.1M | d_model (768) × vocab_size (14407) |
| LayerNorms & Projections | ~1M | - |

Where each transformer layer has: `4×d_model² + 2×d_model×d_ff` parameters.

## Model Capabilities

### 1. Exact Name Queries
```python
query = "function calculate_sum"
# Returns: calculate_sum implementation
```

### 2. Name Decomposition
```python
query = "get user data"
# Returns: get_user_data implementation
```

### 3. Semantic Queries
```python
query = "add two numbers"
# Returns: function with matching docstring
```

## Checkpoint Format

Models are saved in **MLX-compatible NumPy format** (`.npz`). For PyTorch compatibility, safetensors export is supported.

```
checkpoints/malm/
├── config.json      # Model architecture config
├── model.npz        # Model weights (MLX format)
├── tokenizer.json   # Vocabulary
└── functions.json   # Function metadata index
```

## Usage on Your Own Codebase

```python
from code_llm.malm import MALM, Tokenizer, extract_functions, load_model
import mlx.core as mx

# Load pre-trained model
model, tokenizer, _ = load_model("checkpoints/malm")

# Extract functions from your code
with open("your_code.py") as f:
    code = f.read()
functions = extract_functions(code)

# Build memory bank
keys = [tokenizer.add_token(f["name"]) for f in functions]
values = []
for f in functions:
    ids = tokenizer.encode(f["source"])
    ids = ids[:100] + [0] * (100 - len(ids))
    values.append(ids)

keys = mx.array(keys)
values = mx.array(values)
key_emb, val_emb = model.encode_memory(keys, values)

# Query
query = "function that handles authentication"
query_ids = tokenizer.encode(query)
query_ids = query_ids[:20] + [0] * (20 - len(query_ids))
query_ids = mx.array([query_ids])

query_emb = model.encode_query(query_ids)
_, attn, _ = model.retrieve(query_emb, key_emb, val_emb)

# Get best match
best_idx = int(mx.argmax(attn[0]))
print(f"Found: {functions[best_idx]['name']}")
print(functions[best_idx]['source'])
```

## Key Insight

> **MALM achieves 100% exact retrieval because each function name is a single token.**
>
> This is the same principle that makes HashHop work. When keys are single tokens, the model learns a perfect hash function through contrastive training.

## Training

```bash
# Full training (2000 functions, 10K steps, ~20 minutes)
poetry run python code_llm/malm.py \
    --max-functions 2000 \
    --steps 10000 \
    --batch-size 32 \
    --lr 3e-4

# Quick test (100 functions, 1K steps, ~2 minutes)
poetry run python code_llm/malm.py \
    --max-functions 100 \
    --steps 1000
```

## Files

| File | Description |
|------|-------------|
| `malm.py` | Complete MALM implementation (model, tokenizer, training) |
