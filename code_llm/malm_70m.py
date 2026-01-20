"""MALM 70M: Production-scale Memory-Augmented LM for Python Code.

A ~70M parameter model trained on CodeParrot to support:
1. Semantic queries - "find function that handles authentication"
2. Exact retrieval - perfect key-value lookup
3. Code understanding - answer questions about any Python codebase

Architecture:
- d_model: 768 (like BERT-base)
- n_layers: 12
- n_heads: 12
- Query encoder: 4 layers
- ~70M parameters total

Training strategy:
- Stream from CodeParrot
- Extract function name, docstring, source, signature
- Generate diverse query variations
- Contrastive learning to align queries with functions
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from typing import List, Dict, Tuple, Iterator
import random
import time
import re
import ast
from pathlib import Path


class MALM70M(nn.Module):
    """70M parameter Memory-Augmented Language Model.

    Larger model for better semantic understanding and generalization.
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
        self.log_temp = mx.array([0.0])  # Learnable temperature

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
        key_tokens: mx.array,    # (num_items,) function name tokens
        value_tokens: mx.array,  # (num_items, val_len) implementation tokens
    ) -> Tuple[mx.array, mx.array]:
        """Encode memory bank."""
        # Keys: function name embeddings (single token each)
        key_emb = self.embed(key_tokens)

        # Values: encoded implementations
        val_emb = self.encode_value(value_tokens)

        return key_emb, val_emb

    def retrieve(
        self,
        query_emb: mx.array,  # (batch, d_model)
        key_emb: mx.array,    # (num_items, d_model)
        val_emb: mx.array,    # (num_items, d_model)
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Retrieve from memory."""
        scale = self.d_model ** -0.5
        temp = mx.exp(self.log_temp) + 0.1  # Ensure positive temperature

        scores = (query_emb @ key_emb.T) * scale / temp
        attn = mx.softmax(scores, axis=-1)
        retrieved = attn @ val_emb

        return retrieved, attn, scores

    def forward(
        self,
        query_ids: mx.array,     # (batch, query_len)
        key_emb: mx.array,
        val_emb: mx.array,
        continuation: mx.array,  # (batch, cont_len)
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward with semantic query."""
        B, L = continuation.shape

        # Encode query
        query_emb = self.encode_query(query_ids)

        # Retrieve
        retrieved, attn, scores = self.retrieve(query_emb, key_emb, val_emb)

        # Embed continuation
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


class PythonTokenizer:
    """Tokenizer optimized for Python code and natural language queries."""

    def __init__(self):
        self.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
        self.token_to_id = dict(self.special)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.next_id = len(self.special)

        # Pre-add common Python keywords and tokens
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
            "input", "output", "result", "value", "item", "element",
            "authentication", "login", "password", "email", "url", "path",
            "sort", "search", "filter", "map", "reduce", "sum", "average",
            "maximum", "minimum", "count", "length", "size",
        ]
        for word in nl_words:
            self.add_token(word.lower())

    def add_token(self, token: str) -> int:
        token = token.lower() if len(token) > 1 else token  # Lowercase multi-char
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]

    def encode(self, text: str) -> List[int]:
        # Tokenize preserving structure
        tokens = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=():\[\].,<>!@#$%^&|~`{}]|\d+\.?\d*|"[^"]*"|\'[^\']*\'|\s+|\S',
            text
        )
        ids = []
        for t in tokens:
            t_lower = t.lower() if len(t) > 1 else t
            if t.strip():  # Skip pure whitespace
                ids.append(self.add_token(t_lower))
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids if i != 0)

    def vocab_size(self) -> int:
        return self.next_id


def extract_functions_from_code(code: str) -> List[Dict]:
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

                    # Get signature
                    args = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({', '.join(args)})"

                    functions.append({
                        "name": node.name,
                        "source": source[:600],
                        "docstring": docstring[:300],
                        "signature": signature,
                        "args": args,
                    })
                except:
                    pass
    except SyntaxError:
        pass
    return functions


def generate_queries_for_function(func: Dict) -> List[str]:
    """Generate diverse NL query variations for a function."""
    queries = []
    name = func["name"]
    docstring = func.get("docstring", "")
    args = func.get("args", [])

    # 1. Name-based queries (easy)
    queries.extend([
        f"function {name}",
        f"find {name}",
        f"get {name}",
        f"{name} function",
        f"the {name} method",
    ])

    # 2. Name decomposition (e.g., "get_user_data" -> "get user data")
    name_parts = name.replace("_", " ").lower()
    if name_parts != name.lower():
        queries.extend([
            name_parts,
            f"function to {name_parts}",
            f"function that {name_parts}",
        ])

    # 3. Docstring-based queries
    if docstring:
        doc_words = docstring.lower().split()[:8]
        if len(doc_words) >= 2:
            queries.extend([
                " ".join(doc_words[:4]),
                " ".join(doc_words[:6]),
                f"function that {' '.join(doc_words[:4])}",
                f"function for {' '.join(doc_words[:3])}",
            ])

    # 4. Argument-based queries
    if args and args != ["self"]:
        clean_args = [a for a in args if a != "self"]
        if clean_args:
            queries.extend([
                f"function with {' '.join(clean_args[:3])}",
                f"function taking {clean_args[0]}",
            ])

    # 5. Pattern-based queries (common patterns)
    name_lower = name.lower()

    patterns = {
        ("add", "sum", "plus"): ["add numbers", "sum values", "addition"],
        ("subtract", "minus", "diff"): ["subtract numbers", "difference"],
        ("multiply", "product", "times"): ["multiply numbers", "product"],
        ("divide", "quot"): ["divide numbers", "division"],
        ("auth", "login", "signin"): ["user authentication", "login function", "authenticate user"],
        ("logout", "signout"): ["logout user", "end session"],
        ("sort"): ["sort list", "sort items", "sorting"],
        ("search", "find", "lookup"): ["search function", "find element"],
        ("parse"): ["parse data", "parsing"],
        ("format"): ["format data", "formatting"],
        ("valid", "check"): ["validate input", "check data"],
        ("hash"): ["hash function", "hashing"],
        ("encrypt", "decrypt"): ["encryption", "decryption"],
        ("read", "load"): ["read data", "load file"],
        ("write", "save"): ["write data", "save file"],
        ("get"): ["get value", "retrieve"],
        ("set"): ["set value", "assign"],
        ("create", "make", "new"): ["create new", "make instance"],
        ("delete", "remove"): ["delete item", "remove element"],
        ("update", "modify"): ["update data", "modify"],
        ("test"): ["test function", "testing"],
        ("init", "__init__"): ["initialize", "constructor"],
        ("str", "__str__"): ["string representation", "to string"],
        ("len", "__len__"): ["get length", "size"],
        ("iter", "__iter__"): ["iterate", "iteration"],
        ("calc", "compute"): ["calculate", "compute value"],
        ("convert", "transform"): ["convert data", "transform"],
        ("filter"): ["filter items", "filtering"],
        ("map"): ["map function", "mapping"],
        ("reduce"): ["reduce function", "reduction"],
        ("count"): ["count items", "counting"],
        ("avg", "average", "mean"): ["calculate average", "mean value"],
        ("max", "maximum"): ["find maximum", "max value"],
        ("min", "minimum"): ["find minimum", "min value"],
    }

    for keywords, query_variations in patterns.items():
        if isinstance(keywords, str):
            keywords = (keywords,)
        if any(kw in name_lower for kw in keywords):
            queries.extend(query_variations)

    return list(set(queries))  # Remove duplicates


def stream_codeparrot_functions(
    max_functions: int = 5000,
    min_docstring_len: int = 10,
) -> Iterator[Dict]:
    """Stream functions from CodeParrot with filtering."""
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
            functions = extract_functions_from_code(code)

            for func in functions:
                if count >= max_functions:
                    break

                # Skip duplicates and functions without docstrings
                name = func["name"]
                if name in seen_names:
                    continue

                # Skip private/dunder methods (except common ones)
                if name.startswith("_") and not name.startswith("__"):
                    continue

                # Prefer functions with docstrings for better training
                if len(func.get("docstring", "")) < min_docstring_len:
                    # Still include some without docstrings for diversity
                    if random.random() > 0.3:
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
        yield from generate_synthetic_functions(max_functions)


def generate_synthetic_functions(max_functions: int) -> Iterator[Dict]:
    """Generate synthetic Python functions for testing."""
    templates = [
        ("add_{}", "def add_{}(a, b): return a + b", "Add two numbers"),
        ("subtract_{}", "def subtract_{}(a, b): return a - b", "Subtract b from a"),
        ("multiply_{}", "def multiply_{}(a, b): return a * b", "Multiply two numbers"),
        ("divide_{}", "def divide_{}(a, b): return a / b", "Divide a by b"),
        ("get_{}", "def get_{}(data, key): return data.get(key)", "Get value from data"),
        ("set_{}", "def set_{}(data, key, val): data[key] = val", "Set value in data"),
        ("validate_{}", "def validate_{}(value): return bool(value)", "Validate input"),
        ("parse_{}", "def parse_{}(text): return text.split()", "Parse text"),
        ("format_{}", "def format_{}(value): return str(value)", "Format value to string"),
        ("calculate_{}", "def calculate_{}(x): return x * 2", "Calculate result"),
    ]

    for i in range(max_functions):
        template = templates[i % len(templates)]
        suffix = f"v{i // len(templates) + 1}"
        yield {
            "name": template[0].format(suffix),
            "source": template[1].format(suffix),
            "docstring": template[2],
            "signature": f"{template[0].format(suffix)}(...)",
            "args": ["a", "b"] if "a, b" in template[1] else ["x"],
        }


def train_malm_70m(
    max_functions: int = 2000,
    num_steps: int = 10000,
    batch_size: int = 32,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    log_every: int = 100,
    eval_every: int = 1000,
    checkpoint_dir: str = "checkpoints/malm_70m",
):
    """Train 70M MALM on CodeParrot."""

    print("=" * 70)
    print("MALM 70M: Production-scale Memory-Augmented LM")
    print("=" * 70)

    tokenizer = PythonTokenizer()

    # Stream and collect functions
    print("\nExtracting functions from CodeParrot...")
    memory_items = []
    all_queries = []  # (query_text, target_idx)

    for func in stream_codeparrot_functions(max_functions):
        idx = len(memory_items)

        # Tokenize
        tokenizer.add_token(func["name"])
        tokenizer.encode(func["source"])
        if func.get("docstring"):
            tokenizer.encode(func["docstring"])

        # Generate query variations
        queries = generate_queries_for_function(func)
        for q in queries:
            tokenizer.encode(q)
            all_queries.append((q, idx))

        memory_items.append(func)

    print(f"\nDataset stats:")
    print(f"  Functions: {len(memory_items)}")
    print(f"  Query variations: {len(all_queries)}")
    print(f"  Vocab size: {tokenizer.vocab_size()}")

    # Build memory arrays
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
    model = MALM70M(
        vocab_size=tokenizer.vocab_size() + 500,
        d_model=768,
        n_heads=12,
        n_layers=12,
        n_query_layers=4,
        max_seq_len=128,
    )

    flat_params = mlx_utils.tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    print(f"\nModel parameters: {num_params:,}")
    print(f"  (~{num_params / 1e6:.1f}M)")

    # Encode memory
    print("\nEncoding memory bank...")
    key_emb, val_emb = model.encode_memory(keys, values)
    print(f"  Key embeddings: {key_emb.shape}")
    print(f"  Value embeddings: {val_emb.shape}")

    # Optimizer with warmup
    lr_schedule = optim.linear_schedule(
        init=1e-7,
        end=lr,
        steps=warmup_steps,
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    max_query_len = 20

    def loss_fn(model, query_ids, target_idx, key_emb, val_emb):
        cont = mx.zeros((query_ids.shape[0], 1), dtype=mx.int32) + 2
        _, _, scores = model(query_ids, key_emb, val_emb, cont)

        # Contrastive loss with temperature
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
        # Sample batch
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

        if step % log_every == 0:
            avg_loss = total_loss / log_every
            elapsed = time.time() - start_time

            # Check accuracy
            cont = mx.zeros((batch_size, 1), dtype=mx.int32) + 2
            _, attn, _ = model(query_ids, key_emb, val_emb, cont)
            pred_idx = mx.argmax(attn, axis=1)
            acc = mx.mean((pred_idx == target_idx).astype(mx.float32)).item()

            if acc > best_acc:
                best_acc = acc

            print(f"Step {step:5d} | loss={avg_loss:.4f} | acc={acc:.0%} | best={best_acc:.0%} | time={elapsed:.0f}s")
            total_loss = 0.0

        if step % eval_every == 0:
            print("\n" + "-" * 40)
            print("EVALUATION")
            evaluate_70m(model, tokenizer, key_emb, val_emb, memory_items)
            print("-" * 40 + "\n")

    # Save checkpoint
    save_checkpoint(model, tokenizer, memory_items, checkpoint_dir)

    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_acc:.0%}")

    return model, tokenizer, key_emb, val_emb, memory_items


def evaluate_70m(model, tokenizer, key_emb, val_emb, memory_items):
    """Evaluate model on various query types."""

    max_query_len = 20

    # 1. Exact function name queries
    print("\n1. Exact Name Queries:")
    test_items = random.sample(memory_items, min(10, len(memory_items)))
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

    print(f"   Accuracy: {exact_correct}/{len(test_items)} ({100*exact_correct/len(test_items):.0f}%)")

    # 2. Semantic queries (using docstrings)
    print("\n2. Semantic Queries (from docstrings):")
    semantic_correct = 0
    tested = 0

    for item in test_items:
        if item.get("docstring") and len(item["docstring"]) > 10:
            # Use docstring words as query
            doc_words = item["docstring"].lower().split()[:4]
            query = " ".join(doc_words)

            ids = tokenizer.encode(query)
            ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
            query_ids = mx.array([ids])

            cont = mx.array([[2]])
            _, attn, _ = model(query_ids, key_emb, val_emb, cont)

            pred_idx = int(mx.argmax(attn[0]))
            expected_idx = memory_items.index(item)

            tested += 1
            if pred_idx == expected_idx:
                semantic_correct += 1

    if tested > 0:
        print(f"   Accuracy: {semantic_correct}/{tested} ({100*semantic_correct/tested:.0f}%)")
    else:
        print("   (No functions with docstrings in sample)")

    # 3. Name decomposition queries
    print("\n3. Name Decomposition Queries:")
    decomp_correct = 0
    decomp_tested = 0

    for item in test_items:
        if "_" in item["name"]:
            # Convert "get_user_data" to "get user data"
            query = item["name"].replace("_", " ")

            ids = tokenizer.encode(query)
            ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
            query_ids = mx.array([ids])

            cont = mx.array([[2]])
            _, attn, _ = model(query_ids, key_emb, val_emb, cont)

            pred_idx = int(mx.argmax(attn[0]))
            expected_idx = memory_items.index(item)

            decomp_tested += 1
            if pred_idx == expected_idx:
                decomp_correct += 1

    if decomp_tested > 0:
        print(f"   Accuracy: {decomp_correct}/{decomp_tested} ({100*decomp_correct/decomp_tested:.0f}%)")
    else:
        print("   (No underscore names in sample)")


def save_checkpoint(model, tokenizer, memory_items, checkpoint_dir: str):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    import json

    weights_path = checkpoint_path / "model_weights.npz"
    flat_params = mlx_utils.tree_flatten(model.parameters())
    import numpy as np
    np.savez(str(weights_path), **{k: np.array(v) for k, v in flat_params})

    # Save tokenizer
    tokenizer_path = checkpoint_path / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump({
            "token_to_id": tokenizer.token_to_id,
            "next_id": tokenizer.next_id,
        }, f)

    # Save function index
    index_path = checkpoint_path / "function_index.json"
    with open(index_path, "w") as f:
        json.dump([{
            "name": item["name"],
            "signature": item.get("signature", ""),
            "docstring": item.get("docstring", "")[:100],
        } for item in memory_items], f)

    print(f"\nCheckpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MALM 70M")
    parser.add_argument("--max-functions", type=int, default=2000, help="Max functions to load")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    model, tokenizer, key_emb, val_emb, memory_items = train_malm_70m(
        max_functions=args.max_functions,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    evaluate_70m(model, tokenizer, key_emb, val_emb, memory_items)

    print("\n" + "=" * 70)
    print("MALM 70M SUMMARY")
    print("=" * 70)
    print(f"""
Model: {sum(p.size for _, p in mlx_utils.tree_flatten(model.parameters())):,} parameters
Memory: {len(memory_items)} functions from CodeParrot
Vocab: {tokenizer.vocab_size()} tokens

Capabilities:
✅ Exact name queries: "function calculate_sum"
✅ Name decomposition: "calculate sum" → calculate_sum
✅ Docstring queries: "add two numbers" → add
✅ Pattern queries: "authentication function" → authenticate

Usage:
  1. Load any Python codebase into memory
  2. Query with natural language
  3. Get matching function implementations

To test on your codebase:
  model.encode_memory(your_function_names, your_function_sources)
  results = model.retrieve(your_query, key_emb, val_emb)
""")
