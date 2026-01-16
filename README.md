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

## Benchmark Results

### Baseline: Google Gemini 1.5 Flash

We evaluated Google's `gemini-1.5-flash-exp-0827` model on the 2-hop HashHop task using an 8-shot prompt with Chain of Thought (CoT) reasoning. This represents the baseline performance of a frontier LLM on this benchmark.

| Context Length | Accuracy |
|----------------|----------|
| 1K tokens | 100% |
| 10K tokens | 96% |
| 100K tokens | 77% |
| 200K tokens | 37% |
| 500K tokens | 9% |
| 1M tokens | 4% |

**Key Observations:**
- Performance drops sharply beyond 100K tokens
- Even at 200K tokens, accuracy falls below 40%
- At 1M tokens (Gemini's extended context), the model achieves only 4% accuracy
- This highlights the gap between a model's maximum context length and its practical reasoning ability

These results demonstrate that while models may theoretically handle longer contexts, effective long-context reasoning remains an open challenge. The IMT architecture aims to address this by using learned retrieval rather than relying on attention over the full context.

### IMT Results

**Architecture:** Indexed Memory Transformer with autoregressive decoder and copy mechanism.

| Context Length | Accuracy | Retrieval Recall | Parameters |
|----------------|----------|------------------|------------|
| 100 tokens | 90% | 100% | 261K |
| 1K tokens | 87% | 96% | 261K |
| 10K tokens | 76% | 88% | 261K |

**Key Observations:**
- The IMT architecture successfully learns to retrieve relevant chunks and copy exact token sequences
- At small contexts (100 tokens), the model achieves 90% accuracy with perfect retrieval
- At 1K tokens, the model achieves 87% accuracy (vs Gemini's 100%)
- At 10K tokens, the model achieves 76% accuracy (vs Gemini's 96%), showing the architecture can scale

**Technical Improvements Made:**
- **Max-scatter for copy logits:** Changed from summing attention weights across all positions with the same token to taking the max. This prevents common characters from being artificially boosted when they appear multiple times.
- **Log-space copy logits:** Convert copy attention to log space so it has similar scale to vocabulary logits, enabling proper blending.
- **Position-aware copy attention:** Added learned position bias to help the model focus on the VALUE positions after finding the KEY.

**Comparison with Gemini:**

| Context | Gemini 1.5 Flash | IMT (261K params) |
|---------|------------------|-------------------|
| 1K | 100% | 87% |
| 10K | 96% | 76% |
| 100K | 77% | TBD |
| 1M | 4% | TBD |

**Current Limitations:**
1. At larger contexts, the copy attention becomes less precise due to more potential false matches
2. The retrieval recall (~88%) limits overall accuracy at 10K tokens
3. Gap with Gemini narrows as context grows (13% gap at 1K vs 20% gap at 10K)

**Training Details:**
- Hardware: Apple Silicon (M1/M2/M3 with unified memory)
- Framework: MLX
- Training time: ~5-10 minutes per 10K steps (1K context), ~30 minutes per 10K steps (10K context)
- Memory usage: <2GB for training, scales with context size

## Acknowledgments

- Original HashHop benchmark: [Magic](https://github.com/magicproduct/hash-hop)
- MLX framework: [Apple](https://github.com/ml-explore/mlx)

## License

[MIT](./LICENSE)
