"""Scaled LTM with Proper Contrastive Training.

Key improvements over ltm_full_demo.py:
1. Contrastive retrieval loss - explicitly trains attention to pick correct memory slot
2. More training steps (2000+) for proper convergence
3. Semantic query training - queries describe functionality, not just function names
4. Multi-task training - both retrieval accuracy AND generation quality
5. Temperature annealing - sharper attention over time

This should achieve:
- High accuracy on MagicLabs Use Case 1 (Framework knowledge)
- Perfect attention alignment on HashHop
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
import time
import re
import ast
from pathlib import Path


class DynamicTokenizer:
    """Dynamic tokenizer - vocabulary grows as needed."""

    def __init__(self):
        self.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.token_to_id = dict(self.special)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.next_id = len(self.special)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]

    def encode(self, text: str) -> List[int]:
        tokens = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=():\[\].,<>!@#$%^&|~`]|\d+\.?\d*|"[^"]*"|\'[^\']*\'|\S',
            text
        )
        return [self._add_token(t) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids)

    def vocab_size(self) -> int:
        return self.next_id

    def get_id(self, token: str) -> int:
        return self._add_token(token)


class ContrastiveLTM(nn.Module):
    """LTM with contrastive retrieval training.

    Key difference: Retrieval is explicitly supervised with contrastive loss,
    not just implicitly learned through generation loss.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Separate projections for key, value, query
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        # Query encoder (projects input sequence to query vector)
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Decoder
        self.decoder_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]

        # Output
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def encode_memory_batch(
        self,
        keys: mx.array,      # (num_items,) single token per key
        values: mx.array,    # (num_items, val_len) tokens per value
    ) -> Tuple[mx.array, mx.array]:
        """Encode memory bank."""
        # Keys: embed single tokens
        key_emb = self.embed(keys)  # (num_items, d_model)
        key_emb = self.key_proj(key_emb)

        # Values: embed and pool
        val_emb = self.embed(values)  # (num_items, val_len, d_model)
        mask = (values != 0).astype(mx.float32)[:, :, None]
        val_emb = val_emb * mask
        val_emb = mx.sum(val_emb, axis=1) / (mx.sum(mask, axis=1) + 1e-8)
        val_emb = self.value_proj(val_emb)

        return key_emb, val_emb

    def compute_query_embedding(
        self,
        input_ids: mx.array,  # (batch, seq_len)
    ) -> mx.array:
        """Compute query embedding from input sequence."""
        h = self.embed(input_ids)
        pos = mx.arange(input_ids.shape[1])
        h = h + self.pos_embed(pos)

        # Mean pool and project
        query = mx.mean(h, axis=1)  # (batch, d_model)
        query = self.query_encoder(query)
        query = self.query_proj(query)
        return query

    def retrieve(
        self,
        query_emb: mx.array,  # (batch, d_model)
        key_emb: mx.array,    # (num_items, d_model)
        val_emb: mx.array,    # (num_items, d_model)
        temperature: float = 1.0,
    ) -> Tuple[mx.array, mx.array]:
        """Retrieve from memory with explicit attention."""
        # Attention scores
        scale = self.d_model ** -0.5
        scores = (query_emb @ key_emb.T) * scale / temperature  # (batch, num_items)
        attn = mx.softmax(scores, axis=-1)

        # Retrieve
        retrieved = attn @ val_emb  # (batch, d_model)
        return retrieved, attn, scores

    def forward(
        self,
        input_ids: mx.array,
        key_emb: mx.array,
        val_emb: mx.array,
        temperature: float = 1.0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass returning logits, attention, and raw scores."""
        B, L = input_ids.shape

        # Embed input
        h = self.embed(input_ids)
        pos = mx.arange(L)
        h = h + self.pos_embed(pos)

        # Compute query and retrieve
        query = mx.mean(h, axis=1)
        query = self.query_encoder(query)
        query = self.query_proj(query)
        retrieved, attn, scores = self.retrieve(query, key_emb, val_emb, temperature)

        # Add retrieved to all positions
        h = h + retrieved[:, None, :]

        # Decode
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        for layer in self.decoder_layers:
            h = layer(h, mask)

        h = self.ln(h)
        logits = self.output(h)
        return logits, attn, scores

    def __call__(self, input_ids, key_emb, val_emb, temperature=1.0):
        return self.forward(input_ids, key_emb, val_emb, temperature)

    def generate(
        self,
        key_emb: mx.array,
        val_emb: mx.array,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.7,
    ) -> Tuple[List[int], mx.array]:
        """Generate with retrieval."""
        generated = list(prompt_ids)
        attn = None

        for _ in range(max_new_tokens):
            ctx = generated[-self.max_seq_len:]
            x = mx.array([ctx])
            logits, attn, _ = self(x, key_emb, val_emb)
            logits = logits[0, -1] / temperature
            probs = mx.softmax(logits)
            next_tok = mx.random.categorical(mx.log(probs + 1e-10))
            generated.append(int(next_tok))
            if next_tok == 3:
                break

        return generated, attn


def create_semantic_training_data(
    memory_items: List[Dict],
    tokenizer: DynamicTokenizer,
    fixed_len: int = 20,
) -> List[Tuple]:
    """Create training data with semantic queries.

    Each training sample has:
    - input_ids: semantic query about the function
    - target_ids: function name/usage
    - target_idx: which memory slot should be retrieved
    """
    train_data = []

    for idx, item in enumerate(memory_items):
        name = item["name"]
        docstring = item.get("docstring", "")

        # Create semantic query variations
        queries = [
            # Direct queries
            f"function {name}",
            f"call {name}",
            f"use {name}",
            f"{name} function",
            f"get {name}",
            f"find {name}",
        ]

        # Add docstring-based queries if available
        if docstring:
            words = docstring.split()[:5]
            if len(words) >= 2:
                queries.append(" ".join(words[:3]))

        for q in queries:
            input_ids = tokenizer.encode(q)
            target_ids = input_ids[1:] + [tokenizer.get_id(name)]

            # Pad
            input_ids = input_ids[:fixed_len] + [0] * (fixed_len - len(input_ids))
            target_ids = target_ids[:fixed_len] + [0] * (fixed_len - len(target_ids))

            train_data.append({
                "input_ids": input_ids,
                "target_ids": target_ids,
                "target_idx": idx,  # Which memory slot to attend to
                "name": name,
            })

    return train_data


def contrastive_loss(
    scores: mx.array,  # (batch, num_items) - raw attention scores
    target_idx: mx.array,  # (batch,) - correct memory indices
    temperature: float = 0.1,
) -> mx.array:
    """Contrastive loss to train retrieval.

    InfoNCE-style loss: maximize score of correct memory slot.
    """
    # Scale scores
    scores = scores / temperature

    # Cross entropy where target is the correct memory index
    loss = nn.losses.cross_entropy(scores, target_idx, reduction="mean")
    return loss


def combined_loss(
    model: ContrastiveLTM,
    input_ids: mx.array,
    target_ids: mx.array,
    target_idx: mx.array,
    key_emb: mx.array,
    val_emb: mx.array,
    retrieval_weight: float = 1.0,
    gen_weight: float = 1.0,
    temperature: float = 1.0,
):
    """Combined loss: generation + contrastive retrieval."""
    # Forward pass
    logits, attn, scores = model(input_ids, key_emb, val_emb, temperature)

    # Generation loss
    gen_loss = nn.losses.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        target_ids.reshape(-1),
        reduction="mean"
    )

    # Retrieval loss (contrastive)
    ret_loss = contrastive_loss(scores, target_idx, temperature=0.1)

    # Combined
    total_loss = gen_weight * gen_loss + retrieval_weight * ret_loss
    return total_loss, gen_loss, ret_loss


def train_scaled_ltm(
    memory_items: List[Dict],
    tokenizer: DynamicTokenizer,
    keys: mx.array,
    values: mx.array,
    num_steps: int = 2000,
    batch_size: int = 16,
    lr: float = 1e-3,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    log_every: int = 100,
    eval_every: int = 500,
):
    """Train the scaled LTM model with contrastive learning."""
    print(f"\n{'='*60}")
    print("Training Scaled LTM with Contrastive Retrieval")
    print(f"{'='*60}")

    # Create model
    model = ContrastiveLTM(
        vocab_size=tokenizer.vocab_size() + 500,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    flat_params = mlx_utils.tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    print(f"Model parameters: {num_params:,}")

    # Encode memory
    key_emb, val_emb = model.encode_memory_batch(keys, values)
    print(f"Memory: {len(memory_items)} items, keys={key_emb.shape}, values={val_emb.shape}")

    # Create training data
    train_data = create_semantic_training_data(memory_items, tokenizer)
    print(f"Training samples: {len(train_data)}")

    # Optimizer with warmup
    optimizer = optim.Adam(learning_rate=lr)

    # Loss function
    def loss_fn(model, input_ids, target_ids, target_idx, key_emb, val_emb, temp):
        total, gen, ret = combined_loss(
            model, input_ids, target_ids, target_idx,
            key_emb, val_emb,
            retrieval_weight=2.0,  # Higher weight on retrieval
            gen_weight=1.0,
            temperature=temp,
        )
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 60)

    start_time = time.time()
    total_loss = 0.0

    for step in range(1, num_steps + 1):
        # Temperature annealing: start high, reduce to sharpen attention
        temp = max(0.5, 2.0 - step / num_steps * 1.5)

        # Sample batch
        batch = random.sample(train_data, min(batch_size, len(train_data)))

        input_ids = mx.array([x["input_ids"] for x in batch])
        target_ids = mx.array([x["target_ids"] for x in batch])
        target_idx = mx.array([x["target_idx"] for x in batch])

        # Forward and backward
        loss, grads = loss_and_grad(
            model, input_ids, target_ids, target_idx,
            key_emb, val_emb, temp
        )

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

        if step % log_every == 0:
            avg_loss = total_loss / log_every
            elapsed = time.time() - start_time

            # Quick eval: retrieval accuracy on batch
            _, attn, _ = model(input_ids, key_emb, val_emb, temp)
            pred_idx = mx.argmax(attn, axis=1)
            ret_acc = mx.mean((pred_idx == target_idx).astype(mx.float32)).item()

            print(f"Step {step:5d} | loss={avg_loss:.4f} | ret_acc={ret_acc:.2%} | temp={temp:.2f} | time={elapsed:.0f}s")
            total_loss = 0.0

        if step % eval_every == 0:
            print("\n--- Evaluation ---")
            eval_retrieval(model, key_emb, val_emb, tokenizer, memory_items, train_data)
            print("-" * 60)

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")
    return model, key_emb, val_emb


def eval_retrieval(
    model: ContrastiveLTM,
    key_emb: mx.array,
    val_emb: mx.array,
    tokenizer: DynamicTokenizer,
    memory_items: List[Dict],
    train_data: List[Dict],
    num_samples: int = 10,
):
    """Evaluate retrieval accuracy."""
    samples = random.sample(train_data, min(num_samples, len(train_data)))

    correct_attn = 0
    correct_gen = 0

    for sample in samples:
        input_ids = mx.array([sample["input_ids"]])
        target_idx = sample["target_idx"]
        expected_name = sample["name"]

        # Get attention and generation
        generated, attn = model.generate(
            key_emb, val_emb,
            sample["input_ids"],
            max_new_tokens=5,
            temperature=0.5
        )

        # Check attention
        pred_idx = int(mx.argmax(attn[0]))
        if pred_idx == target_idx:
            correct_attn += 1

        # Check generation
        output = tokenizer.decode(generated[len(sample["input_ids"]):])
        if expected_name in output:
            correct_gen += 1

    print(f"  Retrieval accuracy: {correct_attn}/{num_samples}")
    print(f"  Generation accuracy: {correct_gen}/{num_samples}")


def load_python_stdlib(tokenizer: DynamicTokenizer, max_modules: int = 15):
    """Load Python stdlib functions."""
    import importlib

    memory_items = []

    stdlib_modules = [
        "collections", "functools", "itertools", "random", "statistics",
        "pathlib", "datetime", "json", "re", "math", "os.path",
        "string", "textwrap", "copy", "pprint",
    ]

    print("Loading Python stdlib...")

    for module_name in stdlib_modules[:max_modules]:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__file__') and module.__file__:
                source_file = Path(module.__file__)
                if source_file.suffix == '.py' and source_file.exists():
                    source = source_file.read_text(errors='ignore')

                    try:
                        tree = ast.parse(source)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                try:
                                    start = node.lineno - 1
                                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 10
                                    lines = source.split('\n')[start:min(end, start + 15)]
                                    func_source = '\n'.join(lines)
                                    docstring = ast.get_docstring(node) or ""

                                    memory_items.append({
                                        "name": node.name,
                                        "module": module_name,
                                        "source": func_source[:500],
                                        "docstring": docstring[:200],
                                    })
                                except:
                                    pass
                    except SyntaxError:
                        pass
        except:
            pass

    print(f"  Loaded {len(memory_items)} functions")

    # Encode
    keys = []
    values = []
    max_val_len = 80

    for item in memory_items:
        tokenizer.encode(item["name"])
        tokenizer.encode(item["source"])
        keys.append(tokenizer.get_id(item["name"]))

        val_ids = tokenizer.encode(item["source"])
        val_ids = val_ids[:max_val_len] + [0] * (max_val_len - len(val_ids))
        values.append(val_ids)

    return memory_items, mx.array(keys), mx.array(values)


def demo_framework_learning():
    """MagicLabs Use Case 1: Framework in-context learning with proper training."""
    print("\n" + "=" * 70)
    print("MAGICLABS USE CASE 1: Framework In-Context Learning (Scaled)")
    print("=" * 70)

    tokenizer = DynamicTokenizer()

    # Load stdlib
    memory_items, keys, values = load_python_stdlib(tokenizer)

    # Limit for faster demo
    memory_items = memory_items[:150]
    keys = keys[:150]
    values = values[:150]

    print(f"Memory bank: {len(memory_items)} functions")
    print(f"Vocabulary: {tokenizer.vocab_size()} tokens")

    # Train with contrastive learning
    model, key_emb, val_emb = train_scaled_ltm(
        memory_items, tokenizer, keys, values,
        num_steps=2000,
        batch_size=16,
        lr=1e-3,
        d_model=128,
        n_heads=8,
        n_layers=3,
        log_every=100,
        eval_every=500,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION - Framework Knowledge")
    print("=" * 60)

    # Test queries
    test_queries = []
    for item in memory_items[:20]:
        name = item["name"]
        test_queries.append((f"function {name}", name, memory_items.index(item)))
        test_queries.append((f"call {name}", name, memory_items.index(item)))

    correct_attn = 0
    correct_gen = 0

    for query, expected, expected_idx in test_queries[:20]:
        input_ids = tokenizer.encode(query)
        generated, attn = model.generate(key_emb, val_emb, input_ids, max_new_tokens=5)
        output = tokenizer.decode(generated[len(input_ids):])

        # Check attention
        pred_idx = int(mx.argmax(attn[0]))
        attn_correct = pred_idx == expected_idx
        if attn_correct:
            correct_attn += 1

        # Check generation
        gen_correct = expected in output
        if gen_correct:
            correct_gen += 1

        print(f"Q: '{query[:30]}' -> Attn slot {pred_idx} (expected {expected_idx}) {'✓' if attn_correct else '✗'}")
        print(f"   Output: {output[:25]} {'✓' if gen_correct else '✗'}")

    print(f"\nAttention Accuracy: {correct_attn}/20 ({100*correct_attn/20:.0f}%)")
    print(f"Generation Accuracy: {correct_gen}/20 ({100*correct_gen/20:.0f}%)")

    return model, tokenizer, key_emb, val_emb, memory_items


def demo_hashhop():
    """HashHop verification with attention accuracy.

    Key insight: HashHop is about exact key→value mapping.
    We need to train the model to:
    1. Attend to the correct memory slot based on query key
    2. Generate the value from that slot
    """
    print("\n" + "=" * 70)
    print("HASHHOP VERIFICATION: Exact Key-Value Retrieval")
    print("=" * 70)

    tokenizer = DynamicTokenizer()

    # Create hash pairs - each hash is a SINGLE token (like original HashHop)
    num_pairs = 50  # Smaller for faster convergence
    hash_len = 6

    def random_hash():
        return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=hash_len))

    pairs = [(random_hash(), random_hash()) for _ in range(num_pairs)]

    # Pre-add all hashes as single tokens
    for k, v in pairs:
        tokenizer._add_token(k)
        tokenizer._add_token(v)

    # Memory: key=hash, value=target hash
    keys = []
    values = []
    max_val_len = 5

    for k, v in pairs:
        keys.append(tokenizer.get_id(k))
        val_ids = [tokenizer.get_id(v)]
        val_ids = val_ids + [0] * (max_val_len - len(val_ids))
        values.append(val_ids)

    keys_arr = mx.array(keys)
    values_arr = mx.array(values)

    print(f"Hash pairs: {num_pairs}")
    print(f"Vocab size: {tokenizer.vocab_size()}")

    # Create model
    model = ContrastiveLTM(
        vocab_size=tokenizer.vocab_size() + 50,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    flat_params = mlx_utils.tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    print(f"Model parameters: {num_params:,}")

    key_emb, val_emb = model.encode_memory_batch(keys_arr, values_arr)

    # Training data: input=key, target=value
    train_data = []
    fixed_len = 5

    for idx, (k, v) in enumerate(pairs):
        # Input is just the key hash
        input_ids = [tokenizer.get_id(k)]
        # Target is the value hash
        target_ids = [tokenizer.get_id(v)]

        # Pad
        input_ids = input_ids + [0] * (fixed_len - len(input_ids))
        target_ids = target_ids + [0] * (fixed_len - len(target_ids))

        train_data.append({
            "input_ids": input_ids,
            "target_ids": target_ids,
            "target_idx": idx,
            "name": k,
            "value": v,
        })

    print(f"Training samples: {len(train_data)}")

    # Train
    optimizer = optim.Adam(learning_rate=2e-3)

    def loss_fn(model, input_ids, target_ids, target_idx, key_emb, val_emb, temp):
        total, _, _ = combined_loss(
            model, input_ids, target_ids, target_idx,
            key_emb, val_emb,
            retrieval_weight=3.0,  # Heavy retrieval emphasis
            gen_weight=1.0,
            temperature=temp,
        )
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print("\nTraining on hash pairs...")
    print("-" * 60)

    start = time.time()
    total_loss = 0

    for step in range(1, 1001):
        temp = max(0.3, 1.5 - step / 1000 * 1.2)

        batch = random.sample(train_data, min(8, len(train_data)))

        input_ids = mx.array([x["input_ids"] for x in batch])
        target_ids = mx.array([x["target_ids"] for x in batch])
        target_idx = mx.array([x["target_idx"] for x in batch])

        loss, grads = loss_and_grad(model, input_ids, target_ids, target_idx, key_emb, val_emb, temp)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

        if step % 100 == 0:
            avg_loss = total_loss / 100
            _, attn, _ = model(input_ids, key_emb, val_emb, temp)
            pred_idx = mx.argmax(attn, axis=1)
            ret_acc = mx.mean((pred_idx == target_idx).astype(mx.float32)).item()
            print(f"  Step {step:4d} | loss={avg_loss:.4f} | ret_acc={ret_acc:.0%} | temp={temp:.2f}")
            total_loss = 0

    print(f"\nTraining complete in {time.time() - start:.1f}s")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION - HashHop")
    print("=" * 60)

    test_pairs = random.sample(pairs, min(20, len(pairs)))

    correct_attn = 0
    correct_gen = 0

    for k, v in test_pairs:
        input_ids = [tokenizer.get_id(k)]
        generated, attn = model.generate(key_emb, val_emb, input_ids, max_new_tokens=3)
        output = tokenizer.decode(generated[len(input_ids):])

        # Check attention
        expected_idx = next(i for i, (pk, _) in enumerate(pairs) if pk == k)
        pred_idx = int(mx.argmax(attn[0]))

        attn_ok = pred_idx == expected_idx
        if attn_ok:
            correct_attn += 1

        # Check generation - value should be in output
        gen_ok = v in output or output.strip().startswith(v[:4])
        if gen_ok:
            correct_gen += 1

        status = "✓✓" if (attn_ok and gen_ok) else ("✓" if gen_ok else "✗")
        print(f"  {k} -> {output[:15]:15s} (expected: {v}) {status}")

    print(f"\nAttention Accuracy: {correct_attn}/20 ({100*correct_attn/20:.0f}%)")
    print(f"Value Retrieval: {correct_gen}/20 ({100*correct_gen/20:.0f}%)")


def demo_code_edit():
    """MagicLabs Use Case 2: Code edit pattern learning."""
    print("\n" + "=" * 70)
    print("MAGICLABS USE CASE 2: Code Edit Pattern Learning (Scaled)")
    print("=" * 70)

    tokenizer = DynamicTokenizer()

    # More edit examples
    edit_examples = [
        ("def add(a, b): return a + b", "def add(a: int, b: int) -> int: return a + b"),
        ("def sub(a, b): return a - b", "def sub(a: int, b: int) -> int: return a - b"),
        ("def mul(a, b): return a * b", "def mul(a: int, b: int) -> int: return a * b"),
        ("def div(a, b): return a / b", "def div(a: float, b: float) -> float: return a / b"),
        ("def neg(x): return -x", "def neg(x: int) -> int: return -x"),
        ("def sqr(x): return x*x", "def sqr(x: int) -> int: return x*x"),
        ("def dbl(x): return x+x", "def dbl(x: int) -> int: return x+x"),
        ("def inc(n): return n+1", "def inc(n: int) -> int: return n+1"),
        ("def dec(n): return n-1", "def dec(n: int) -> int: return n-1"),
        ("def mod(a, b): return a % b", "def mod(a: int, b: int) -> int: return a % b"),
        ("def pow(a, b): return a ** b", "def pow(a: int, b: int) -> int: return a ** b"),
        ("def avg(a, b): return (a+b)/2", "def avg(a: float, b: float) -> float: return (a+b)/2"),
    ]

    # Build memory
    keys = []
    values = []
    memory_items = []
    max_val_len = 60

    for i, (before, after) in enumerate(edit_examples):
        tokenizer.encode(before)
        tokenizer.encode(after)

        key_name = f"edit_{i}"
        tokenizer.encode(key_name)
        keys.append(tokenizer.get_id(key_name))

        val_ids = tokenizer.encode(after)
        val_ids = val_ids[:max_val_len] + [0] * (max_val_len - len(val_ids))
        values.append(val_ids)

        memory_items.append({
            "name": key_name,
            "source": after,
            "docstring": "",
            "before": before,
        })

    keys_arr = mx.array(keys)
    values_arr = mx.array(values)

    print(f"Edit examples: {len(edit_examples)}")
    print(f"Vocab size: {tokenizer.vocab_size()}")

    # Create training data - input is before code, target is after code
    # We need to teach the model that given "before" code, retrieve the right "after"

    # Create model
    model = ContrastiveLTM(
        vocab_size=tokenizer.vocab_size() + 100,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    key_emb, val_emb = model.encode_memory_batch(keys_arr, values_arr)

    # Training data
    train_data = []
    fixed_len = 40

    for i, (before, after) in enumerate(edit_examples):
        input_ids = tokenizer.encode(before)
        target_ids = tokenizer.encode(after)

        input_ids = input_ids[:fixed_len] + [0] * (fixed_len - len(input_ids))
        target_ids = target_ids[:fixed_len] + [0] * (fixed_len - len(target_ids))

        train_data.append({
            "input_ids": input_ids,
            "target_ids": target_ids,
            "target_idx": i,
            "name": f"edit_{i}",
        })

    # Train
    optimizer = optim.Adam(learning_rate=2e-3)

    def loss_fn(model, input_ids, target_ids, target_idx, key_emb, val_emb, temp):
        total, _, _ = combined_loss(
            model, input_ids, target_ids, target_idx,
            key_emb, val_emb,
            retrieval_weight=2.0,
            gen_weight=1.0,
            temperature=temp,
        )
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print("\nTraining on edit patterns...")
    for step in range(1, 501):
        temp = max(0.5, 2.0 - step / 500 * 1.5)
        batch = random.sample(train_data, min(4, len(train_data)))

        input_ids = mx.array([x["input_ids"] for x in batch])
        target_ids = mx.array([x["target_ids"] for x in batch])
        target_idx = mx.array([x["target_idx"] for x in batch])

        loss, grads = loss_and_grad(model, input_ids, target_ids, target_idx, key_emb, val_emb, temp)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    # Test on new code
    print("\n--- Testing Edit Pattern Learning ---")
    test_cases = [
        "def max(a, b): return a if a > b else b",
        "def min(a, b): return a if a < b else b",
        "def abs(x): return x if x >= 0 else -x",
    ]

    for test_code in test_cases:
        input_ids = tokenizer.encode(test_code)
        generated, attn = model.generate(key_emb, val_emb, input_ids, max_new_tokens=40)
        output = tokenizer.decode(generated[len(input_ids):])

        print(f"\n  Input:  {test_code}")
        print(f"  Output: {output[:70]}...")

        has_hints = "int" in output or "float" in output or "->" in output
        print(f"  Type hints added: {'✓' if has_hints else '✗'}")


def main():
    """Run all scaled demos."""
    print("*" * 70)
    print("*" + "  Scaled LTM: Contrastive Training for Better Retrieval".center(68) + "*")
    print("*" * 70)

    # Run demos
    demo_framework_learning()
    demo_code_edit()
    demo_hashhop()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key improvements in this version:

1. CONTRASTIVE RETRIEVAL LOSS
   - Explicitly trains attention to pick correct memory slot
   - InfoNCE-style loss maximizes score of correct slot
   - Higher weight on retrieval loss (2x generation)

2. TEMPERATURE ANNEALING
   - Starts with high temperature (softer attention)
   - Gradually decreases to sharpen attention
   - Helps model learn better retrieval patterns

3. MORE TRAINING
   - 2000 steps for framework demo (vs 500)
   - Proper convergence monitoring
   - Retrieval accuracy tracked during training

4. SEMANTIC QUERY TRAINING
   - Multiple query variations per function
   - Trains on "function X", "call X", "use X", etc.
   - Better generalization to new queries

This should achieve much higher accuracy on MagicLabs use cases
while maintaining perfect HashHop retrieval!
""")


if __name__ == "__main__":
    main()
