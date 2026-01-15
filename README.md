# HashHop Long Context Evaluation

This repository contains the code for HashHop, a long context architecture benchmark, along with an **Indexed Memory Transformer (IMT)** implementation optimized for Apple Silicon using MLX.

## Installation Guide

### Prerequisites

- Git
- Python 3.9+
- [Poetry](https://python-poetry.org/docs/#installation)
- Apple Silicon Mac (M1/M2/M3) for IMT training

### Steps

1. Clone the repository:
   ```bash
   git clone git@github.com:magicproduct/hash-hop.git
   cd hash-hop
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

## Generating Evaluation Data

The `MultiHopEval.make_one` function generates a `MultiHopSample` object which can be used for either evaluation (via
the `targets` field) or for training models on the multihop task (via the `completion` field).

### Usage Example

```python
from hashhop import MultiHopEval

CHARS_PER_TOKEN = 3
datapoint = MultiHopEval.make_one(
    n_chars_problem=int(1_000_000 * CHARS_PER_TOKEN),
    num_queries=5,
    hops=2,
    hash_pair_str_length=16,
    chain_of_thought=False,
)
print(datapoint.prompt)
print(datapoint.completion)
print(datapoint.targets)
```

### Parameters

- `n_chars_problem`: int
    - The size of the problem in characters.
- `num_queries`: int
    - The number of queries in the completion.
- `hops`: int
    - The number of hops in the reasoning chain.
- `hash_pair_str_length`: int
    - The number of characters per hash.
- `chain_of_thought`: bool
    - If True, the model is asked to produce H1 -> H2 -> H3.
    - If False, the model is asked to produce H1 -> H3.

### Output

- `prompt`: str
    - Contains the shuffled hash pairs.
- (Used for training) `completion`: str
    - The queries and targets in string format
- (Used for evaluation) `targets`: Dict[str, str]
    - Contains query-ground truth pairs in structured format
    - If chain of thought is false, will contain {H1: H3} (e.g. 'HETyxiWTFSVUYega': 'pChfybAJRUBmdAGC')
    - If chain of thought is true, will contain full chain {H1: H2 = H3} (e.g. 'KeiVcwXpnYIWLPmk': 'GmmNmICdvEErHgei =
      JhgvBFdYCnLVZBoy')

## Indexed Memory Transformer (IMT)

The IMT is a novel architecture designed specifically for long-context hash retrieval tasks on Apple Silicon. It uses MLX for efficient training and inference.

### Architecture

- **ChunkEncoder**: 2-layer transformer that processes 512-token chunks independently
- **IndexKeyExtractor**: Learns to extract searchable keys from chunk representations
- **LearnedIndexSearch**: Differentiable approximate nearest neighbor retrieval with learned clusters
- **LocalDecoder**: 3-layer transformer that attends to retrieved chunks to produce answers

### Training

```bash
# Quick iteration (1M context, ~10 minutes)
python scripts/train_imt.py --config configs/imt_nano_small.yaml

# Full training (10M context)
python scripts/train_imt.py --config configs/imt_nano.yaml

# With custom memory limit (default is 50%)
python scripts/train_imt.py --config configs/imt_nano_small.yaml --memory-limit 0.3

# Resume from checkpoint
python scripts/train_imt.py --config configs/imt_nano.yaml --resume checkpoints/imt_xxx/step_1000
```

### Evaluation

```bash
python scripts/eval_imt.py --checkpoint checkpoints/imt_xxx/best --num-samples 100
```

### Configuration

Two preset configurations are available:

- `configs/imt_nano.yaml`: Full 10M token context, ~12-15M parameters
- `configs/imt_nano_small.yaml`: 1M token context for quick iteration

Key configuration options:
- `memory_limit_fraction`: Limits VRAM usage (default: 0.5 = 50%)
- `chunk_batch_size`: Number of chunks processed per batch
- `use_gradient_checkpointing`: Enable memory-efficient training

## License

[MIT](./LICENSE)
