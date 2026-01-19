# HashHop: Solving Long-Context Retrieval at Scale

HashHop is a benchmark for evaluating long-context retrieval capabilities of large language models. It tests an LLM's ability to follow chains of hash-pair associations across long prompts (up to millions of tokens).

This repository provides:
1. **The HashHop Benchmark**: Data generation for multi-hop hash retrieval tasks
2. **Tokenized HashHop Solver**: A simple architecture that achieves **100% accuracy at 1M+ tokens**

> **Note**: This project extends the original [HashHop benchmark by Magic](https://github.com/magicproduct/hash-hop) with our tokenized solution.

## Key Result

| Context Length | Gemini 1.5 Flash | Tokenized Solver |
|----------------|------------------|------------------|
| 1K tokens | 100% | **100%** |
| 10K tokens | 96% | **100%** |
| 100K tokens | 77% | **100%** |
| 1M tokens | 4% | **100%** |

**The tokenized solver achieves 100% accuracy where Gemini 1.5 Flash drops to 4%.**

## Installation

```bash
git clone https://github.com/codelion/hash-hop.git
cd hash-hop
poetry install
```

## Quick Start

```bash
# Run tokenized solver on 10K token context
poetry run python tokenized_hashhop.py --tokens 10000

# Run full benchmark
poetry run python tokenized_hashhop.py --benchmark
```

## The HashHop Task

HashHop tests a model's ability to retrieve information across long contexts by following chains of hash assignments:

```
# Example with 2 hops:
ABCDEFGHIJKLMNOp = QRSTUVWXYZabcdef    # First hop
QRSTUVWXYZabcdef = 'ghijklmnopqrstuv'  # Second hop (final value in quotes)

# Query: ABCDEFGHIJKLMNOp -> Answer: ghijklmnopqrstuv
```

The task becomes harder as:
- Context length increases (more hash pairs to search through)
- Number of hops increases (longer chains to follow)
- Hash strings are random (no semantic patterns)

### Generating Evaluation Data

```python
from hashhop import MultiHopEval

datapoint = MultiHopEval.make_one(
    n_chars_problem=3_000_000,  # ~1M tokens
    num_queries=5,
    hops=2,
    hash_pair_str_length=16,
    chain_of_thought=False,
)
print(datapoint.prompt)      # Shuffled hash pairs
print(datapoint.targets)     # Query -> answer mapping
```

## The Tokenized Solution

### Why It Works

The key insight is that **tokenization converts HashHop into MQAR** (Multi-Query Associative Recall), which transformers solve perfectly via [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/).

| Approach | Challenge | Result |
|----------|-----------|--------|
| Character-level | Must learn "ABCD" == "ABCD" by comparing 4 chars | Fails at scale |
| Token-level | Just match token ID 42 == token ID 42 | Works perfectly |

When each unique hash string becomes a single token:
- No character-level matching needed
- Attention can be sharp (one-hot)
- Induction heads naturally implement key-value lookup

### Architecture

```
Input: "ABCDEFGH = IJKLMNOP"
         ↓
    [Tokenize each hash string as single token]
         ↓
Query Token → Attention over Key Tokens → Retrieve Value Token
         ↓
Output: Predicted hash string
```

The architecture is remarkably simple:
- **Tokenizer**: Maps each unique N-char hash string to a token ID
- **Embedding Layer**: Learned embeddings for each token
- **Attention**: Query-key dot product with hard attention (low temperature)
- **Retrieval**: Attention-weighted value embedding → nearest token lookup

### Usage

```bash
# Basic usage
poetry run python tokenized_hashhop.py --tokens 1000

# Full benchmark across scales
poetry run python tokenized_hashhop.py --benchmark

# Custom parameters
poetry run python tokenized_hashhop.py \
    --tokens 100000 \
    --hash-length 16 \
    --hops 2 \
    --steps 1000 \
    --d-model 128
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tokens` | 1000 | Context size in tokens |
| `--hash-length` | 16 | Characters per hash string |
| `--hops` | 2 | Number of hops to follow |
| `--steps` | 1000 | Training steps |
| `--d-model` | 128 | Embedding dimension |
| `--benchmark` | - | Run full benchmark |

## Benchmark Results

### Comparison with Gemini 1.5 Flash

We evaluated Google's `gemini-1.5-flash-exp-0827` on 2-hop HashHop with 8-shot CoT prompting:

| Context | Gemini 1.5 Flash | Tokenized Solver | Improvement |
|---------|------------------|------------------|-------------|
| 1K tokens | 100% | **100%** | - |
| 10K tokens | 96% | **100%** | +4% |
| 100K tokens | 77% | **100%** | +23% |
| 200K tokens | 37% | **100%** | +63% |
| 500K tokens | 9% | **100%** | +91% |
| 1M tokens | 4% | **100%** | +96% |

### Why Gemini Fails

Gemini's performance degradation reveals a fundamental limitation:
1. **Attention entropy collapse**: Softmax attention spreads across all positions
2. **No exact matching**: Subword tokenization fragments random hash strings
3. **Lost in the middle**: Known issue where LLMs struggle with mid-context retrieval

### Why Tokenization Succeeds

1. **Single-token hashes**: Each hash string is one token, enabling exact matching
2. **Hard attention**: Low temperature makes attention nearly one-hot
3. **Induction heads**: The architecture naturally implements [A][B]...[A] → [B] pattern

This is confirmed by prior research:
- [Zoology paper](https://arxiv.org/abs/2312.04927): Transformers solve MQAR perfectly
- [Induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/): How in-context learning works

## Implications

### For LLM Evaluation

HashHop with random character strings tests a **fundamentally different capability** than real-world retrieval:
- Real tokens have learned embeddings that enable matching
- Random strings require learning embeddings from scratch
- This explains why frontier LLMs struggle despite long context windows

### For Architecture Design

The solution suggests that long-context retrieval benefits from:
1. **Token-level exact matching** (not character-level)
2. **Hard attention** (sparse, not softmax)
3. **Explicit key-value structure** (not implicit in context)

### For Practical Applications

When building retrieval-augmented systems:
- Consider tokenization strategy for retrieval targets
- Use structured key-value formats when possible
- Don't rely solely on LLM context for precise lookups

## Repository Structure

```
hashhop/
  __init__.py        # Exports MultiHopEval
  generate.py        # Benchmark data generation
  test_generate.py   # Unit tests
tokenized_hashhop.py # Tokenized solver implementation
```

## Development

```bash
# Run tests
poetry run pytest

# Code quality
poetry run ruff check --fix
poetry run ruff format
poetry run mypy
```

## Acknowledgments

- Original HashHop benchmark: [Magic](https://github.com/magicproduct/hash-hop)
- Zoology/MQAR research: [Stanford](https://arxiv.org/abs/2312.04927)
- Induction heads research: [Anthropic](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)

## License

[MIT](./LICENSE)
