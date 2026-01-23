# MALM: Memory-Augmented Language Model for Code

MALM is a 165M parameter model for semantic code search, applying the same insight that solved HashHop: treating each function name as a single token enables perfect key-value retrieval.

## Pre-trained Model

A pre-trained MALM model is available on HuggingFace:

```bash
# Download and run inference
pip install mlx huggingface_hub numpy
huggingface-cli download codelion/malm-165m --local-dir ./malm-165m
python malm-165m/inference.py --query "function that sorts a list"
```

**Example output:**

```text
Query: function that sorts a list
------------------------------------------------------------

1. array_sort (score: 0.9526)
   Signature: array_sort(col)
   Docstring: Collection function: sorts the input array in ascending order...

2. sort_array (score: 0.7707)
   Signature: sort_array(col, asc)
   Docstring: Collection function: sorts the input array in ascending or descending order...
```

**Repository:** [`codelion/malm-165m`](https://huggingface.co/codelion/malm-165m)

## Demos: MALM + Qwen2.5-Coder

The `demos/` directory shows MALM retrieval combined with Qwen2.5-Coder for code generation:

```bash
# Run both demos
poetry run python code_llm/demos/run_demo.py
```

### Demo 1: In-Context GUI Framework

Creates a calculator using a **completely novel** GUI framework that the model learns purely from retrieved context:

```bash
poetry run python code_llm/demos/run_demo.py
```

**Example output:**

```text
USE CASE 1: In-Context GUI Framework
Creating a calculator with a NOVEL framework
============================================================

Generated Calculator App:
+------------------------------------------+
| Simple Calculator                        |
| [_____] [_____]                         |
| [=     ]                                 |
| [+] [-] [*] [/] [C]                     |
+------------------------------------------+

--- Live Demo ---
  42 + 8 = 50.0
  100 - 37 = 63.0
  7 * 6 = 42.0
  144 / 12 = 12.0

✓ Calculator works with the custom GUI framework!
```

### Demo 2: Password Strength Meter

Adds a password strength component to a DocuSign-like app by retrieving relevant code:

**Example output:**

```text
USE CASE 2: Real Repo Code Editing
Adding password strength meter to signup page
============================================================

Password: 'P@ssw0rd!' (Strong password)
🟢 [█████] Very Strong
  ✓ At least 8 characters
  ✓ Contains uppercase
  ✓ Contains lowercase
  ✓ Contains number
  ✓ Contains special char

✓ Password strength meter integrates with signup page!
```

### System Architecture (~1.7B total)

```
┌─────────────────────────────────────────────────────────────┐
│  User Query: "Create calculator with CustomGUI framework"   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MALM Retriever (~165M params)                   │
│  • Searches 10M+ token codebase                             │
│  • Returns top-k relevant code chunks                       │
│  • Perfect retrieval via single-token keys                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Qwen2.5-Coder-1.5B (~1.5B params)                  │
│  • Takes query + retrieved context                          │
│  • Generates code following patterns                        │
│  • In-context learning from examples                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      Generated Code
```

> **Key insight**: Small model (~1.7B) + retrieval = frontier-like capabilities on large codebases

## Training Your Own Model

```bash
# Standard training (2K functions, ~20 minutes)
poetry run python code_llm/malm.py --max-functions 2000 --steps 10000

# Large-scale training (20K functions, ~90 minutes)
poetry run python code_llm/malm.py \
    --max-functions 20000 \
    --steps 30000 \
    --batch-size 64 \
    --max-vocab-size 50000
```

## Architecture

MALM is a **165M parameter** memory-augmented transformer:

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
│  Generate output based on retrieved function context         │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Breakdown (165M total)

| Component | Parameters |
|-----------|------------|
| Token Embedding | 11.1M |
| Position Embedding | 0.1M |
| Query Encoder (4 layers) | 28.4M |
| Value Encoder (4 layers) | 28.4M |
| Decoder (12 layers) | 85.1M |
| Output Projection | 11.1M |
| **Total** | **~165M** |

## Results

### Small Scale (2K functions)

| Query Type | Accuracy |
|------------|----------|
| Exact Name Queries | **100%** |
| Semantic Queries | **100%** |

### Large Scale (20K functions)

| Query Type | Accuracy |
|------------|----------|
| Exact Name Queries | **70%** |
| Semantic Queries | **67%** |

## Python API

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Download model
model_path = snapshot_download("codelion/malm-165m")
sys.path.insert(0, model_path)

from inference import load_model, search_functions

# Load
model, tokenizer, functions, config = load_model(Path(model_path))

# Search
results = search_functions(
    model, tokenizer, functions,
    query="connect to database",
    top_k=5
)

for name, signature, docstring, score in results:
    print(f"{name}: {score:.4f}")
```

## Files

| File | Description |
|------|-------------|
| `malm.py` | Complete MALM implementation |
| `demos/run_demo.py` | Interactive demo script |
| `demos/malm_coder_demo.py` | MALM + Qwen integration |
| `demos/custom_gui_framework.py` | Novel GUI framework for demo |
| `demos/docusign_app/` | Sample app for code editing demo |

## Key Insight

> **MALM achieves perfect retrieval because each function name is a single token.**
>
> This is the same principle that makes HashHop work. When keys are single tokens, the model learns a perfect hash function through contrastive training.

## Citation

```bibtex
@article{sharma2026malm,
  title={Reverse Engineering a $500M Mystery: From HashHop to Memory-Augmented Language Models},
  author={Sharma, Asankhaya},
  year={2026},
  url={https://huggingface.co/blog/codelion/reverse-engineering-magic-hashhop}
}
```
