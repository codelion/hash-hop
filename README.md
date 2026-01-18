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

**Architecture:** Indexed Memory Transformer with autoregressive decoder and copy mechanism (pointer network).

The IMT uses a fully generic architecture with no task-specific heuristics:
- Standard transformer encoder for chunk encoding
- Learned dense retrieval (query embedding similarity)
- Standard pointer network for copy mechanism
- No hardcoded knowledge of the KEY>VALUE format

**Current Status:**

| Context Length | Accuracy | Retrieval Recall | Parameters | Training Steps |
|----------------|----------|------------------|------------|----------------|
| 1K tokens | 0% | 99% | 916K | 50K |
| 10K tokens | 0% | 83% | 916K | 100K |

**Key Finding:**
The retrieval component works well (83-99% recall), but the copy mechanism fails to learn which positions to attend to without task-specific guidance.

**The Challenge:**
For HashHop, the model must learn that for query "ABCD", it should:
1. Find where "ABCD" appears in the context
2. Skip the ">" separator
3. Copy the next 4 characters

With pure learned attention over 512-2048 context positions, this pattern is extremely hard to learn from scratch. The copy attention becomes nearly uniform, unable to focus on the correct positions.

**Technical Details:**
- **Max-scatter for copy logits:** Using max instead of sum prevents common tokens from being artificially boosted
- **Log-space copy logits:** Converts attention weights to log space for proper blending with vocab logits
- **Pure learned copy attention:** No position bias or task-specific heuristics

**Comparison with Gemini:**

| Context | Gemini 1.5 Flash | IMT (916K params) |
|---------|------------------|-------------------|
| 1K | 100% | 0% |
| 10K | 96% | 0% |
| 100K | 77% | - |
| 1M | 4% | - |

**Analysis:**
This result demonstrates that the HashHop task requires either:
1. **Much larger models** with more capacity to learn the copy pattern
2. **Much longer training** (millions of steps instead of 100K)
3. **Architectural innovations** like relative positional encoding in the copy mechanism
4. **Pre-training** on related tasks before fine-tuning on HashHop

The retrieval component is effective, but the "last mile" of copying the correct tokens from context remains an open challenge for small models without task-specific inductive biases.

**Training Details:**
- Hardware: Apple Silicon (M1/M2/M3 with unified memory)
- Framework: MLX
- Training time: ~30 minutes per 10K steps
- Memory usage: <2GB for training, scales with context size

### T5-base Fine-tuning Results

We also tested fine-tuning Google's T5-base (220M parameters) on HashHop to see if a pretrained encoder-decoder model could solve the task.

| Context Length | Accuracy | Parameters | Training Steps |
|----------------|----------|------------|----------------|
| 200 chars | **98%** | 220M | 2K |
| 1K chars | 0-2% | 220M | 20K |

**Key Findings:**
- T5-base achieves **98% accuracy** on very short contexts (200 chars, ~20 hash pairs)
- On longer contexts (1K chars, ~100 hash pairs), T5-base struggles even after 20K training steps
- The model learns the output format (4-character strings) but fails at the retrieval

**Why T5 struggles on longer contexts:**
1. **Tokenization mismatch:** T5 uses SentencePiece which fragments random character strings unpredictably
2. **Attention scaling:** Full attention over ~500 tokens is harder than over ~100 tokens
3. **Pattern complexity:** Finding one matching hash among 100+ pairs requires more capacity

### T5-base Curriculum Learning

We implemented curriculum learning to progressively train on harder examples:

| Level | Context | Accuracy | Steps | Status |
|-------|---------|----------|-------|--------|
| 1 | 100 chars | **100%** | 1K | PASS |
| 2 | 200 chars | **100%** | 1K | PASS |
| 3 | 500 chars | **6%** | 3K | FAIL (threshold: 70%) |

**Key Findings:**
- Curriculum learning successfully teaches T5-base to master short contexts (100-200 chars)
- The model achieves 100% accuracy on both Level 1 and Level 2
- **No catastrophic forgetting**: Previous levels maintain 93-100% accuracy during Level 3 training
- However, the model fails to generalize to 500 chars even with curriculum learning
- Loss at Level 3 remains high (~0.7-1.4) compared to near-zero at Levels 1-2

**Training Command:**
```bash
poetry run python t5/train_curriculum.py --max-level 5 --eval-every 200
```

### ByT5 (Byte-level T5) Results

We also tested ByT5-small (300M parameters), a byte-level variant of T5 that processes raw UTF-8 bytes instead of subword tokens. This eliminates tokenization issues with random character strings.

| Context Length | Accuracy | Parameters | Training Steps |
|----------------|----------|------------|----------------|
| 200 chars | **0%** | 300M | 5000 |

**Status:** ByT5-small failed to learn the task even after 5000 steps. Loss remained constant at ~5.75 (near random), indicating a potential architecture/weight loading issue. The byte-level tokenization should theoretically help, but the model is not training properly.

**Training Command:**
```bash
# Install with torch for weight conversion
poetry install -E torch

# Train ByT5-small
poetry run python t5/train_byt5.py --context-size 200 --max-steps 5000 --batch-size 2
```

**Comparison:**

| Context | Gemini 1.5 Flash | T5-base (220M) | T5 + Curriculum | ByT5-small (300M) | IMT (916K) |
|---------|------------------|----------------|-----------------|-------------------|------------|
| 100 chars | 100% | - | **100%** | - | - |
| 200 chars | 100% | 98% | **100%** | 0% | - |
| 500 chars | 100% | - | **6%** | - | - |
| 1K chars | 100% | 0-2% | - | - | 0% |
| 10K chars | 96% | - | - | - | 0% |

**Conclusion:**
- **Curriculum learning helps** but doesn't fully solve the scaling problem
- T5-base can achieve 100% on short contexts (100-200 chars) with curriculum learning
- The jump from 200 to 500 chars proves difficult - accuracy drops from 100% to 6%
- This suggests a fundamental limitation in T5-base's ability to handle longer retrieval tasks
- Next steps: Try T5-large (770M params) or investigate architectural changes

## Acknowledgments

- Original HashHop benchmark: [Magic](https://github.com/magicproduct/hash-hop)
- MLX framework: [Apple](https://github.com/ml-explore/mlx)

## License

[MIT](./LICENSE)
