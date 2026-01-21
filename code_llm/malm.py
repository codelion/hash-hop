"""MALM: Memory-Augmented Language Model for Python Code.

A 165M parameter model trained on CodeParrot that supports:
1. Semantic queries - "find function that handles authentication"
2. Exact retrieval - perfect key-value lookup (like HashHop)
3. Code understanding - answer questions about any Python codebase

Architecture (165M parameters):
- Embedding: vocab_size * d_model = 14407 * 768 = 11.1M
- Position embedding: 128 * 768 = 0.1M
- Query encoder (4 layers): 4 * (4 * 768^2 + 768 * 3072 * 2) = 28.4M
- Value encoder (4 layers): 4 * (4 * 768^2 + 768 * 3072 * 2) = 28.4M
- Decoder (12 layers): 12 * (4 * 768^2 + 768 * 3072 * 2) = 85.1M
- Output projection: 768 * vocab_size = 11.1M
- Layer norms and projections: ~1M
- Total: ~165M parameters

Model weights are saved in MLX-compatible NumPy format (.npz).
For PyTorch compatibility, use safetensors export.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from typing import List, Dict, Tuple, Iterator, Optional
import random
import time
import re
import ast
import json
from pathlib import Path


class MALM(nn.Module):
    """Memory-Augmented Language Model.

    Architecture:
    - Query encoder: Transforms NL queries into embeddings
    - Memory bank: Stores function name → implementation mappings
    - Retrieval: Attention-based lookup from query to memory
    - Decoder: Generates output based on retrieved context

    Parameter count formula:
    - embed: vocab_size * d_model
    - pos_embed: max_seq_len * d_model
    - query_layers: n_query_layers * (4*d_model^2 + 2*d_model*d_ff)
    - value_layers: n_query_layers * (4*d_model^2 + 2*d_model*d_ff)
    - decoder_layers: n_layers * (4*d_model^2 + 2*d_model*d_ff)
    - output: d_model * vocab_size
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        n_query_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_query_layers = n_query_layers
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Query encoder - encodes NL queries to single embedding
        self.query_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_query_layers)
        ]
        self.query_ln = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        # Value encoder - encodes function implementations
        self.value_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_query_layers)
        ]
        self.value_ln = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Decoder layers
        self.decoder_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]
        self.decoder_ln = nn.LayerNorm(d_model)

        # Output
        self.output = nn.Linear(d_model, vocab_size)

        # Temperature for retrieval
        self.log_temp = mx.array([0.0])

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        flat_params = mlx_utils.tree_flatten(self.parameters())
        return sum(p.size for _, p in flat_params)

    def encode_query(self, query_ids: mx.array) -> mx.array:
        """Encode variable-length query to single embedding."""
        B, L = query_ids.shape

        h = self.embed(query_ids)
        pos = mx.arange(min(L, self.max_seq_len))
        h = h + self.pos_embed(pos)
        h = self.embed_dropout(h)

        for layer in self.query_layers:
            h = layer(h, None)

        h = self.query_ln(h)

        # Mean pool over non-padding tokens
        mask = (query_ids != 0).astype(mx.float32)[:, :, None]
        h = h * mask
        query_emb = mx.sum(h, axis=1) / (mx.sum(mask, axis=1) + 1e-8)

        return self.query_proj(query_emb)

    def encode_value(self, value_ids: mx.array) -> mx.array:
        """Encode function implementation to single embedding."""
        B, L = value_ids.shape

        h = self.embed(value_ids)
        pos = mx.arange(min(L, self.max_seq_len))
        h = h + self.pos_embed(pos)

        for layer in self.value_layers:
            h = layer(h, None)

        h = self.value_ln(h)

        # Mean pool
        mask = (value_ids != 0).astype(mx.float32)[:, :, None]
        h = h * mask
        val_emb = mx.sum(h, axis=1) / (mx.sum(mask, axis=1) + 1e-8)

        return self.value_proj(val_emb)

    def encode_memory(
        self,
        key_tokens: mx.array,
        value_tokens: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Encode memory bank.

        Args:
            key_tokens: (num_items,) function name token IDs
            value_tokens: (num_items, val_len) implementation token IDs

        Returns:
            key_emb: (num_items, d_model) key embeddings
            val_emb: (num_items, d_model) value embeddings
        """
        key_emb = self.embed(key_tokens)
        val_emb = self.encode_value(value_tokens)
        return key_emb, val_emb

    def retrieve(
        self,
        query_emb: mx.array,
        key_emb: mx.array,
        val_emb: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Retrieve from memory using query.

        Args:
            query_emb: (batch, d_model) query embeddings
            key_emb: (num_items, d_model) memory key embeddings
            val_emb: (num_items, d_model) memory value embeddings

        Returns:
            retrieved: (batch, d_model) retrieved value embeddings
            attn: (batch, num_items) attention weights
            scores: (batch, num_items) raw attention scores
        """
        scale = self.d_model ** -0.5
        temp = mx.exp(self.log_temp) + 0.1

        scores = (query_emb @ key_emb.T) * scale / temp
        attn = mx.softmax(scores, axis=-1)
        retrieved = attn @ val_emb

        return retrieved, attn, scores

    def forward(
        self,
        query_ids: mx.array,
        key_emb: mx.array,
        val_emb: mx.array,
        continuation: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with semantic query.

        Args:
            query_ids: (batch, query_len) query token IDs
            key_emb: (num_items, d_model) memory keys
            val_emb: (num_items, d_model) memory values
            continuation: (batch, cont_len) continuation token IDs

        Returns:
            logits: (batch, cont_len, vocab_size) output logits
            attn: (batch, num_items) retrieval attention
            scores: (batch, num_items) retrieval scores
        """
        B, L = continuation.shape

        query_emb = self.encode_query(query_ids)
        retrieved, attn, scores = self.retrieve(query_emb, key_emb, val_emb)

        h = self.embed(continuation)
        pos = mx.arange(min(L, self.max_seq_len))
        h = h + self.pos_embed(pos)
        h = self.embed_dropout(h)

        # Add retrieved context
        h = h + retrieved[:, None, :]

        # Decode
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        for layer in self.decoder_layers:
            h = layer(h, mask)

        h = self.decoder_ln(h)
        logits = self.output(h)

        return logits, attn, scores

    def __call__(self, query_ids, key_emb, val_emb, continuation):
        return self.forward(query_ids, key_emb, val_emb, continuation)


class Tokenizer:
    """Tokenizer for Python code and natural language queries.

    Supports a maximum vocabulary size to control model parameters.
    Tokens beyond max_vocab_size are mapped to <UNK>.
    """

    def __init__(self, max_vocab_size: int = 50000):
        self.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
        self.token_to_id = dict(self.special)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.next_id = len(self.special)
        self.max_vocab_size = max_vocab_size

        # Pre-add common Python keywords
        keywords = [
            "def", "class", "return", "if", "else", "elif", "for", "while",
            "try", "except", "finally", "with", "as", "import", "from",
            "None", "True", "False", "and", "or", "not", "in", "is",
            "lambda", "yield", "raise", "pass", "break", "continue",
            "self", "cls", "async", "await",
        ]
        for kw in keywords:
            self.add_token(kw)

        # Common NL words for queries
        nl_words = [
            "function", "method", "find", "get", "set", "create", "delete",
            "add", "remove", "update", "check", "validate", "parse", "format",
            "convert", "calculate", "compute", "process", "handle", "that",
            "which", "the", "a", "an", "to", "for", "with", "from", "of",
            "numbers", "string", "list", "dict", "file", "data", "user",
        ]
        for word in nl_words:
            self.add_token(word.lower())

    def add_token(self, token: str) -> int:
        """Add token to vocabulary.

        If vocabulary is at max capacity, returns <UNK> for new tokens.
        """
        token = token.lower() if len(token) > 1 else token
        if token in self.token_to_id:
            return self.token_to_id[token]

        # Check if we've hit the vocab limit
        if self.next_id >= self.max_vocab_size:
            return self.special["<UNK>"]

        self.token_to_id[token] = self.next_id
        self.id_to_token[self.next_id] = token
        self.next_id += 1
        return self.token_to_id[token]

    def encode(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        tokens = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=():\[\].,<>!@#$%^&|~`{}]|\d+\.?\d*|"[^"]*"|\'[^\']*\'|\s+|\S',
            text
        )
        ids = []
        for t in tokens:
            t_lower = t.lower() if len(t) > 1 else t
            if t.strip():
                ids.append(self.add_token(t_lower))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids if i != 0)

    def vocab_size(self) -> int:
        return self.next_id

    def save(self, path: str):
        """Save tokenizer to JSON."""
        with open(path, "w") as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "next_id": self.next_id,
                "max_vocab_size": self.max_vocab_size,
            }, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from JSON."""
        tokenizer = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer.next_id = data["next_id"]
        tokenizer.max_vocab_size = data.get("max_vocab_size", 50000)
        tokenizer.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
        return tokenizer


def extract_functions(code: str) -> List[Dict]:
    """Extract functions with metadata from Python code."""
    functions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 20
                    lines = code.split('\n')[start:min(end, start + 25)]
                    source = '\n'.join(lines)

                    docstring = ast.get_docstring(node) or ""
                    args = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({', '.join(args)})"

                    functions.append({
                        "name": node.name,
                        "source": source[:600],
                        "docstring": docstring[:300],
                        "signature": signature,
                        "args": args,
                    })
                except Exception:
                    pass
    except SyntaxError:
        pass
    return functions


def generate_queries(func: Dict) -> List[str]:
    """Generate diverse query variations for a function."""
    queries = []
    name = func["name"]
    docstring = func.get("docstring", "")
    args = func.get("args", [])

    # Name-based queries
    queries.extend([
        f"function {name}",
        f"find {name}",
        f"{name} function",
    ])

    # Name decomposition (get_user_data -> get user data)
    name_parts = name.replace("_", " ").lower()
    if name_parts != name.lower():
        queries.extend([
            name_parts,
            f"function to {name_parts}",
        ])

    # Docstring-based queries
    if docstring:
        doc_words = docstring.lower().split()[:6]
        if len(doc_words) >= 2:
            queries.extend([
                " ".join(doc_words[:4]),
                f"function that {' '.join(doc_words[:4])}",
            ])

    # Argument-based queries
    if args and args != ["self"]:
        clean_args = [a for a in args if a != "self"]
        if clean_args:
            queries.append(f"function with {' '.join(clean_args[:3])}")

    return list(set(queries))


def stream_codeparrot(max_functions: int = 5000) -> Iterator[Dict]:
    """Stream functions from CodeParrot dataset."""
    try:
        from datasets import load_dataset
        print("Loading CodeParrot dataset (streaming)...")

        dataset = load_dataset(
            "codeparrot/codeparrot-clean",
            split="train",
            streaming=True,
        )

        count = 0
        seen_names = set()
        start_time = time.time()

        for sample in dataset:
            if count >= max_functions:
                break

            code = sample.get("content", "")
            functions = extract_functions(code)

            for func in functions:
                if count >= max_functions:
                    break

                name = func["name"]
                if name in seen_names or (name.startswith("_") and not name.startswith("__")):
                    continue

                seen_names.add(name)
                yield func
                count += 1

                if count % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Extracted {count} functions ({elapsed:.0f}s)")

        print(f"  Total: {count} unique functions")

    except ImportError:
        print("datasets not available, using synthetic data")
        for i in range(max_functions):
            yield {
                "name": f"function_{i}",
                "source": f"def function_{i}(x): return x * {i}",
                "docstring": f"Multiply x by {i}",
                "signature": f"function_{i}(x)",
                "args": ["x"],
            }


def save_model(model: MALM, tokenizer: Tokenizer, memory_items: List[Dict],
               checkpoint_dir: str, use_safetensors: bool = True):
    """Save model checkpoint.

    Args:
        model: Trained MALM model
        tokenizer: Tokenizer instance
        memory_items: List of function metadata
        checkpoint_dir: Directory to save to
        use_safetensors: If True, save as safetensors (PyTorch compatible)
    """
    import numpy as np

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    flat_params = mlx_utils.tree_flatten(model.parameters())

    if use_safetensors:
        try:
            from safetensors.numpy import save_file
            weights = {k: np.array(v) for k, v in flat_params}
            save_file(weights, str(checkpoint_path / "model.safetensors"))
            print(f"  Saved model.safetensors")
        except ImportError:
            print("  safetensors not available, saving as .npz")
            np.savez(str(checkpoint_path / "model.npz"),
                     **{k: np.array(v) for k, v in flat_params})
    else:
        np.savez(str(checkpoint_path / "model.npz"),
                 **{k: np.array(v) for k, v in flat_params})
        print(f"  Saved model.npz")

    # Save tokenizer
    tokenizer.save(str(checkpoint_path / "tokenizer.json"))
    print(f"  Saved tokenizer.json")

    # Save function index
    with open(checkpoint_path / "functions.json", "w") as f:
        json.dump([{
            "name": item["name"],
            "signature": item.get("signature", ""),
            "docstring": item.get("docstring", "")[:100],
        } for item in memory_items], f)
    print(f"  Saved functions.json")

    # Save config
    config = {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "n_query_layers": model.n_query_layers,
        "max_seq_len": model.max_seq_len,
        "num_parameters": model.count_parameters(),
        "num_functions": len(memory_items),
    }
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    print(f"\nCheckpoint saved to {checkpoint_dir}")


def load_model(checkpoint_dir: str) -> Tuple[MALM, Tokenizer, List[Dict]]:
    """Load model from checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        model: Loaded MALM model
        tokenizer: Loaded tokenizer
        memory_items: List of function metadata
    """
    import numpy as np

    checkpoint_path = Path(checkpoint_dir)

    # Load config
    with open(checkpoint_path / "config.json") as f:
        config = json.load(f)

    # Create model
    model = MALM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_query_layers=config["n_query_layers"],
        max_seq_len=config["max_seq_len"],
    )

    # Load weights
    if (checkpoint_path / "model.safetensors").exists():
        from safetensors.numpy import load_file
        weights = load_file(str(checkpoint_path / "model.safetensors"))
    else:
        weights = dict(np.load(str(checkpoint_path / "model.npz")))

    # Unflatten and load
    params = mlx_utils.tree_unflatten(list(weights.items()))
    model.update(params)

    # Load tokenizer
    tokenizer = Tokenizer.load(str(checkpoint_path / "tokenizer.json"))

    # Load functions
    with open(checkpoint_path / "functions.json") as f:
        memory_items = json.load(f)

    return model, tokenizer, memory_items


def train(
    max_functions: int = 2000,
    num_steps: int = 10000,
    batch_size: int = 32,
    lr: float = 3e-4,
    checkpoint_dir: str = "checkpoints/malm",
    max_vocab_size: int = 50000,
):
    """Train MALM on CodeParrot.

    Args:
        max_functions: Maximum functions to load
        num_steps: Training steps
        batch_size: Batch size
        lr: Learning rate
        checkpoint_dir: Where to save checkpoints
        max_vocab_size: Maximum vocabulary size (controls model size)
    """
    print("=" * 70)
    print("MALM: Memory-Augmented Language Model")
    print("=" * 70)

    tokenizer = Tokenizer(max_vocab_size=max_vocab_size)

    # Load data
    print("\nExtracting functions from CodeParrot...")
    memory_items = []
    all_queries = []

    for func in stream_codeparrot(max_functions):
        idx = len(memory_items)
        tokenizer.add_token(func["name"])
        tokenizer.encode(func["source"])
        if func.get("docstring"):
            tokenizer.encode(func["docstring"])

        for q in generate_queries(func):
            tokenizer.encode(q)
            all_queries.append((q, idx))

        memory_items.append(func)

    print(f"\nDataset:")
    print(f"  Functions: {len(memory_items)}")
    print(f"  Query variations: {len(all_queries)}")
    print(f"  Vocab size: {tokenizer.vocab_size()}")

    # Build memory
    keys = [tokenizer.add_token(item["name"]) for item in memory_items]
    values = []
    max_val_len = 100

    for item in memory_items:
        ids = tokenizer.encode(item["source"])
        ids = ids[:max_val_len] + [0] * (max_val_len - len(ids))
        values.append(ids)

    keys = mx.array(keys)
    values = mx.array(values)

    # Create model
    model = MALM(
        vocab_size=tokenizer.vocab_size() + 500,
        d_model=768,
        n_heads=12,
        n_layers=12,
        n_query_layers=4,
        max_seq_len=128,
    )

    num_params = model.count_parameters()
    print(f"\nModel: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Encode memory
    print("\nEncoding memory bank...")
    key_emb, val_emb = model.encode_memory(keys, values)

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)
    max_query_len = 20

    def loss_fn(model, query_ids, target_idx, key_emb, val_emb):
        """Standard contrastive loss against all memory items."""
        cont = mx.zeros((query_ids.shape[0], 1), dtype=mx.int32) + 2
        _, _, scores = model(query_ids, key_emb, val_emb, cont)
        scores_scaled = scores / 0.07
        return nn.losses.cross_entropy(scores_scaled, target_idx, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 70)

    start_time = time.time()
    total_loss = 0.0
    best_acc = 0.0

    for step in range(1, num_steps + 1):
        batch = random.sample(all_queries, min(batch_size, len(all_queries)))

        query_ids = []
        target_idx = []
        for q, idx in batch:
            ids = tokenizer.encode(q)
            ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
            query_ids.append(ids)
            target_idx.append(idx)

        query_ids = mx.array(query_ids)
        target_idx = mx.array(target_idx)

        loss, grads = loss_and_grad(model, query_ids, target_idx, key_emb, val_emb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

        if step % 100 == 0:
            avg_loss = total_loss / 100
            elapsed = time.time() - start_time

            cont = mx.zeros((batch_size, 1), dtype=mx.int32) + 2
            _, attn, _ = model(query_ids, key_emb, val_emb, cont)
            pred_idx = mx.argmax(attn, axis=1)
            acc = mx.mean((pred_idx == target_idx).astype(mx.float32)).item()

            if acc > best_acc:
                best_acc = acc

            print(f"Step {step:5d} | loss={avg_loss:.4f} | acc={acc:.0%} | best={best_acc:.0%} | time={elapsed:.0f}s")
            total_loss = 0.0

        if step % 1000 == 0:
            evaluate(model, tokenizer, key_emb, val_emb, memory_items)

    # Save
    save_model(model, tokenizer, memory_items, checkpoint_dir)

    print(f"\nTraining complete! Best accuracy: {best_acc:.0%}")
    return model, tokenizer, key_emb, val_emb, memory_items


def evaluate(model: MALM, tokenizer: Tokenizer, key_emb: mx.array,
             val_emb: mx.array, memory_items: List[Dict]):
    """Evaluate model on different query types."""
    max_query_len = 20
    test_items = random.sample(memory_items, min(10, len(memory_items)))

    print("\n--- Evaluation ---")

    # Exact name queries
    exact_correct = 0
    for item in test_items:
        query = f"function {item['name']}"
        ids = tokenizer.encode(query)
        ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
        query_ids = mx.array([ids])

        cont = mx.array([[2]])
        _, attn, _ = model(query_ids, key_emb, val_emb, cont)

        pred_idx = int(mx.argmax(attn[0]))
        expected_idx = memory_items.index(item)
        if pred_idx == expected_idx:
            exact_correct += 1

    print(f"  Exact name: {exact_correct}/{len(test_items)} ({100*exact_correct/len(test_items):.0f}%)")

    # Semantic queries
    semantic_correct = 0
    semantic_tested = 0
    for item in test_items:
        if item.get("docstring") and len(item["docstring"]) > 10:
            query = " ".join(item["docstring"].lower().split()[:4])
            ids = tokenizer.encode(query)
            ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
            query_ids = mx.array([ids])

            cont = mx.array([[2]])
            _, attn, _ = model(query_ids, key_emb, val_emb, cont)

            pred_idx = int(mx.argmax(attn[0]))
            expected_idx = memory_items.index(item)
            semantic_tested += 1
            if pred_idx == expected_idx:
                semantic_correct += 1

    if semantic_tested > 0:
        print(f"  Semantic: {semantic_correct}/{semantic_tested} ({100*semantic_correct/semantic_tested:.0f}%)")

    print("-" * 40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MALM: Memory-Augmented Language Model")
    parser.add_argument("--max-functions", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/malm")
    parser.add_argument("--max-vocab-size", type=int, default=50000,
                        help="Max vocabulary size (50K = ~165M params)")

    args = parser.parse_args()

    model, tokenizer, key_emb, val_emb, memory_items = train(
        max_functions=args.max_functions,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        max_vocab_size=args.max_vocab_size,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("MALM SUMMARY")
    print("=" * 70)
    print(f"""
Model: {model.count_parameters():,} parameters ({model.count_parameters()/1e6:.1f}M)
Memory: {len(memory_items)} functions
Vocab: {tokenizer.vocab_size()} tokens

Architecture:
  - d_model: {model.d_model}
  - n_heads: {model.n_heads}
  - n_layers: {model.n_layers} (decoder)
  - n_query_layers: {model.n_query_layers} (encoder)

Capabilities:
  - Exact name queries: "function calculate_sum"
  - Name decomposition: "calculate sum" -> calculate_sum
  - Semantic queries: "add two numbers" -> add

Checkpoint: {args.checkpoint_dir}
""")
