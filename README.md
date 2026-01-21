# HashHop: From Benchmark to Memory-Augmented LLM

HashHop is a benchmark for evaluating long-context retrieval in large language models, plus a practical demonstration that the same tokenization insight enables building effective memory-augmented models.

## What's in This Repository

| Component | Description | Key Result |
|-----------|-------------|------------|
| **HashHop Benchmark** | Multi-hop hash retrieval task | Tests LLM long-context capabilities |
| **Tokenized Solver** | Simple architecture for HashHop | **100% accuracy at 10M tokens** |
| **[MALM](code_llm/)** | Memory-Augmented Language Model | Semantic code search with 165M params |

> **Key Insight**: Treating each hash string (or function name) as a single token enables perfect key-value retrieval through attention.

## The HashHop Benchmark

HashHop tests a model's ability to follow chains of hash assignments across long contexts:

```
# Example with 2 hops:
ABCDEFGHIJKLMNOp = QRSTUVWXYZabcdef    # First hop
QRSTUVWXYZabcdef = 'ghijklmnopqrstuv'  # Second hop (final value in quotes)

# Query: ABCDEFGHIJKLMNOp -> Answer: ghijklmnopqrstuv
```

### Benchmark Results

| Context Length | Gemini 1.5 Flash | Tokenized Solver |
|----------------|------------------|------------------|
| 1K tokens | 100% | **100%** |
| 10K tokens | 96% | **100%** |
| 100K tokens | 77% | **100%** |
| 1M tokens | 4% | **100%** |
| 10M tokens | - | **100%** |

The tokenized solver achieves 100% accuracy where frontier LLMs fail because it treats each hash string as a single token, converting the problem to Multi-Query Associative Recall (MQAR).

## From HashHop to MALM

The same tokenization principle that solves HashHop also enables practical memory-augmented models:

```
HashHop:  hash_string → hash_string → final_value
MALM:     function_name → function_implementation
```

**[MALM (Memory-Augmented Language Model)](code_llm/)** applies this insight to code retrieval:
- 165M parameter model for semantic code search
- Pre-trained on CodeParrot dataset
- Available on HuggingFace: [`codelion/malm-165m`](https://huggingface.co/codelion/malm-165m)

See the [code_llm/](code_llm/) directory for full documentation and demos.

## Installation

```bash
git clone https://github.com/codelion/hash-hop.git
cd hash-hop
poetry install
```

## Quick Start

### HashHop Benchmark

```bash
# Run tokenized solver on 10K token context
poetry run python tokenized_hashhop.py --tokens 10000

# Run full benchmark (1K to 10M tokens)
poetry run python tokenized_hashhop.py --benchmark
```

### MALM Code Search

```bash
# Download pre-trained model and run inference
pip install mlx huggingface_hub numpy
huggingface-cli download codelion/malm-165m --local-dir ./malm-165m
python malm-165m/inference.py --query "function that sorts a list"
```

Or train your own:

```bash
poetry run python code_llm/malm.py --max-functions 2000 --steps 10000
```

## The Tokenization Insight

### Why It Works

| Approach | Challenge | Result |
|----------|-----------|--------|
| Character-level | Must learn "ABCD" == "ABCD" by comparing 4 chars | Fails at scale |
| Token-level | Just match token ID 42 == token ID 42 | Works perfectly |

With whole-string tokenization:
1. **Single-token keys**: Each hash/function name is one token
2. **Random orthogonality**: High-dimensional random embeddings are nearly orthogonal
3. **Hard attention**: Low temperature makes attention nearly one-hot

### Architecture

```
Input: "ABCDEFGH = IJKLMNOP"
         ↓
    [Tokenize each string as single token]
         ↓
Query Token → Attention over Key Tokens → Retrieve Value Token
         ↓
Output: Matched value
```

## Generating Evaluation Data

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

## Repository Structure

```
hashhop/
  __init__.py           # Exports MultiHopEval
  generate.py           # Benchmark data generation
  test_generate.py      # Unit tests
tokenized_hashhop.py    # Tokenized solver implementation
code_llm/
  README.md             # MALM documentation
  malm.py               # Memory-Augmented LM implementation
  demos/                # MALM + Qwen demos
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
