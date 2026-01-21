# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HashHop is a long-context evaluation benchmark for large language models. It tests an LLM's ability to follow chains of hash-pair associations across long prompts (up to millions of tokens).

This repository includes:
1. **HashHop Benchmark**: Data generation for multi-hop hash retrieval tasks
2. **Tokenized Solver**: A simple architecture achieving 100% accuracy at 1M+ tokens

## Repository Structure

```
hashhop/
  __init__.py           # Exports MultiHopEval
  generate.py           # Core evaluation generation logic
  test_generate.py      # Unit tests for generation
tokenized_hashhop.py    # Tokenized solver implementation
tests/                  # Test suite
```

## Key Commands

### Setup
```bash
poetry install
```

### Running the Solver
```bash
# Basic usage
poetry run python tokenized_hashhop.py --tokens 10000

# Full benchmark
poetry run python tokenized_hashhop.py --benchmark
```

### Running Tests
```bash
poetry run pytest
```

### Code Quality
```bash
poetry run ruff check --fix   # Linting
poetry run ruff format        # Formatting
poetry run mypy               # Type checking
```

## Architecture

### HashHop Benchmark (`hashhop/generate.py`)

The core class is `MultiHopEval`:

- **`MultiHopEval.make_one()`**: Generates a single evaluation sample with:
  - `n_chars_problem`: Total prompt size in characters
  - `num_queries`: Number of queries in the completion
  - `hops`: Number of association hops to follow (chain length)
  - `hash_pair_str_length`: Characters per hash string
  - `chain_of_thought`: Whether to require intermediate steps in output

- **`MultiHopSample`**: Dataclass containing:
  - `prompt`: Shuffled hash pairs (e.g., `ABC = DEF`, `DEF = 'GHI'`)
  - `completion`: Expected model output format
  - `targets`: Query-to-answer mapping for evaluation

### Tokenized Solver (`tokenized_hashhop.py`)

Key insight: Treat each hash string as a single token (MQAR approach).

Components:
- **HashTokenizer**: Maps hash strings to unique token IDs
- **TokenizedRetriever**: Learned embeddings with hard attention
- **Multi-hop following**: Iteratively retrieves through chain

## Code Style

- Python 3.9+ with type annotations
- Line length: 100 characters
- Uses ruff for linting and formatting
- Uses mypy for type checking
