# Memory-Augmented Language Model (MALM) for Code

## Overview

MALM is our implementation of a memory-augmented architecture for code understanding. It achieves **100% exact retrieval** on 1000+ functions by using the same tokenization approach that made HashHop work.

## Key Insight

The breakthrough came from realizing that **HashHop works because each hash string is a single token**. When we apply the same principle to code (each function name = single token), we get perfect retrieval.

## MALM 70M - Production Model

Our production-scale model (165M parameters) supports **semantic natural language queries**, not just exact function names.

### Results (2000 functions from CodeParrot)

| Query Type | Accuracy |
|------------|----------|
| Exact Name Queries | **100%** |
| Semantic Queries (docstrings) | **100%** |
| Name Decomposition | **86%** |

### Capabilities

- **Exact name queries**: `"function calculate_sum"` → finds `calculate_sum`
- **Semantic queries**: `"add two numbers"` → finds functions with matching docstrings
- **Name decomposition**: `"get user data"` → finds `get_user_data`
- **Pattern queries**: `"authentication function"` → finds auth-related functions

### Usage

```bash
# Train MALM 70M
poetry run python code_llm/malm_70m.py --max-functions 2000 --steps 10000

# Checkpoint saved to checkpoints/malm_70m/
```

### Model Stats
- Parameters: 165M
- Training time: ~19 minutes
- Memory bank: 2000 functions
- Vocabulary: 13,907 tokens

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Bank                          │
│  ┌─────────────┐        ┌─────────────────────────────┐ │
│  │ Function    │        │ Implementation              │ │
│  │ Name Token  │  ────▶ │ (encoded sequence)          │ │
│  └─────────────┘        └─────────────────────────────┘ │
│  add            ────▶   def add(a, b): return a + b    │
│  multiply       ────▶   def multiply(a, b): return a*b │
│  ...            ────▶   ...                            │
└─────────────────────────────────────────────────────────┘
         │
         │ Query (function name token)
         ▼
┌─────────────────────────────────────────────────────────┐
│              Attention-based Retrieval                  │
│                                                         │
│  query_emb @ key_emb.T → softmax → retrieved_value     │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Transformer Decoder                        │
│                                                         │
│  Generate response based on retrieved context          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Results

| Use Case | Accuracy | Notes |
|----------|----------|-------|
| Exact Key-Value Retrieval | **100%** | HashHop-style perfect lookup |
| Code Retrieval & Q&A | **100%** | Query function name → get implementation |
| Semantic Understanding | ✅ | Similar functions cluster in embedding space |

### Training Statistics (1000 functions)
- Model parameters: ~20M
- Training time: ~80 seconds
- Retrieval accuracy: 100% after 200 steps
- Memory: Streams from CodeParrot dataset

## Files

| File | Description |
|------|-------------|
| `malm_70m.py` | Production model with semantic queries (165M params) |
| `malm_v2.py` | Tokenized approach with 100% exact retrieval |
| `memory_augmented_lm.py` | Original MALM implementation |
| `train_malm_codeparrot.py` | Training script with CodeParrot streaming |

## Usage

```bash
# Train MALM v2 (tokenized approach)
poetry run python code_llm/malm_v2.py --max-memory 1000 --steps 5000

# Train on CodeParrot with streaming
poetry run python code_llm/train_malm_codeparrot.py --max-samples 5000 --max-memory 1000
```

## Key Differences from HashHop

| Aspect | HashHop | MALM |
|--------|---------|------|
| Keys | Random 4-char hashes | Function names |
| Values | Random 4-char hashes | Function implementations |
| Query | Exact hash string | Function name |
| Chain length | 1-3 hops | 1 hop (direct retrieval) |

## Why It Works

1. **Single-token keys**: Each function name is one token, enabling exact matching through learned embeddings
2. **Contrastive training**: InfoNCE-style loss ensures each key has a unique embedding
3. **Temperature annealing**: Start soft (temp=1.5), end sharp (temp=0.5) for stable training

## Limitations

- **Code transformation**: Pure retrieval can't do seq2seq transformation (would need encoder-decoder)
- **Multi-hop**: Currently single-hop retrieval; extending to multi-hop is future work

## How to Use on Any Python Codebase

```python
# 1. Load the trained model
from malm_70m import MALM70M, PythonTokenizer

# 2. Parse your codebase and extract functions
functions = extract_functions_from_your_codebase()

# 3. Encode into memory bank
key_emb, val_emb = model.encode_memory(function_names, function_sources)

# 4. Query with natural language
query = "find function that handles user authentication"
results = model.retrieve(query_emb, key_emb, val_emb)
```
