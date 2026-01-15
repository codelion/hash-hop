# HashHop Long Context Evaluation

HashHop is a benchmark for evaluating long-context retrieval capabilities of large language models. It tests an LLM's ability to follow chains of hash-pair associations across long prompts (up to millions of tokens).

This repository includes:
1. **The HashHop Benchmark**: Data generation for multi-hop hash retrieval tasks
2. **Indexed Memory Transformer (IMT)**: A novel architecture optimized for Apple Silicon using MLX

> **Note**: This project is based on the original [HashHop benchmark by Magic](https://github.com/magicproduct/hash-hop). We have extended it with our own IMT implementation and benchmark results.

## Installation

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/docs/#installation)
- Apple Silicon Mac (M1/M2/M3) for IMT training

### Setup

```bash
git clone https://github.com/codelion/hash-hop.git
cd hash-hop
poetry install
```

## The HashHop Benchmark

HashHop evaluates a model's ability to retrieve information across long contexts by following chains of hash assignments. For example, given thousands of assignments like `H1 = H2`, `H2 = H3`, `H3 = 'answer'`, the model must follow the chain from a query hash to find the final quoted value.

### Generating Evaluation Data

```python
from hashhop import MultiHopEval

datapoint = MultiHopEval.make_one(
    n_chars_problem=1_000_000,  # ~1M tokens
    num_queries=5,
    hops=2,
    hash_pair_str_length=16,
    chain_of_thought=False,
)
print(datapoint.prompt)      # Shuffled hash pairs
print(datapoint.targets)     # Query -> answer mapping for evaluation
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `n_chars_problem` | Total prompt size in characters |
| `num_queries` | Number of queries to answer |
| `hops` | Chain length (number of hops to follow) |
| `hash_pair_str_length` | Characters per hash string |
| `chain_of_thought` | If True, output includes intermediate steps |

## Indexed Memory Transformer (IMT)

The IMT is our novel architecture designed specifically for long-context retrieval tasks. Instead of using full attention over millions of tokens, it:

1. **Chunks** the context into manageable 512-token segments
2. **Indexes** each chunk with learned keys for fast retrieval
3. **Retrieves** only the relevant chunks for each query
4. **Decodes** the answer using local attention over retrieved chunks

This approach enables efficient training and inference on Apple Silicon with configurable memory limits.

### Architecture

- **ChunkEncoder**: 2-layer transformer processing 512-token chunks
- **IndexKeyExtractor**: Learns searchable keys from chunk representations
- **LearnedIndexSearch**: Differentiable approximate nearest neighbor retrieval
- **LocalDecoder**: 3-layer transformer attending to retrieved chunks

### Training

```bash
# Quick iteration (1M context)
python scripts/train_imt.py --config configs/imt_nano_small.yaml

# Full training (10M context)
python scripts/train_imt.py --config configs/imt_nano.yaml

# Limit GPU memory to 30% (default is 50%)
python scripts/train_imt.py --config configs/imt_nano_small.yaml --memory-limit 0.3

# Resume from checkpoint
python scripts/train_imt.py --config configs/imt_nano.yaml --resume checkpoints/imt_xxx/step_1000
```

### Evaluation

```bash
python scripts/eval_imt.py --checkpoint checkpoints/imt_xxx/best --num-samples 100
```

### Configuration

| Config | Context Size | Parameters | Use Case |
|--------|-------------|------------|----------|
| `imt_nano_small.yaml` | 1M tokens | ~12-15M | Quick iteration |
| `imt_nano.yaml` | 10M tokens | ~12-15M | Full training |

Key options:
- `memory_limit_fraction`: VRAM limit (default: 0.5 = 50%)
- `chunk_batch_size`: Chunks per encoding batch
- `use_gradient_checkpointing`: Memory-efficient training

## Results

*Coming soon: Benchmark results comparing IMT against baseline approaches.*

## Acknowledgments

- Original HashHop benchmark: [Magic](https://github.com/magicproduct/hash-hop)
- MLX framework: [Apple](https://github.com/ml-explore/mlx)

## License

[MIT](./LICENSE)
