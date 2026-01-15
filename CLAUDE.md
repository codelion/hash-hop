# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HashHop is a long-context evaluation benchmark for large language models, developed by Magic. It tests an LLM's ability to follow chains of hash-pair associations across long prompts (up to millions of tokens).

## Repository Structure

```
hashhop/
  __init__.py      # Exports MultiHopEval
  generate.py      # Core evaluation generation logic
  test_generate.py # Unit tests for generation
test.py            # Additional test file
train.py           # Training utilities
```

## Key Commands

### Setup
```bash
poetry install
```

### Running Tests
```bash
poetry run pytest
```

### Code Quality (Pre-commit hooks)
```bash
poetry run ruff check --fix   # Linting
poetry run ruff format        # Formatting
poetry run mypy               # Type checking
poetry run codespell .        # Spell checking
```

### Run Pre-commit Manually
```bash
poetry run pre-commit run --all-files
```

## Architecture

The core class is `MultiHopEval` in `hashhop/generate.py`:

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

The benchmark works by creating chains of hash associations where the model must follow N hops to find the final value (marked with quotes).

## Code Style

- Python 3.9+ with type annotations required (`disallow_untyped_defs = true`)
- Line length: 100 characters
- Uses ruff for linting and formatting
- Uses mypy for type checking
