"""LTM v2: Simpler architecture closer to HashHop success.

Key insight from HashHop: The retrieval works because each token has a unique
embedding, and attention naturally retrieves the right one.

For code LLM, we need:
1. Memory keys: function names as unique tokens
2. Memory values: function implementations
3. Query: user's question
4. Retrieval: query attends to memory keys, retrieves values
5. Generation: LLM generates based on retrieved values

This is like HashHop but with semantic queries instead of exact match.

IMPORTANT: Uses dynamic tokenization - any Python code works, symbols
are added to vocabulary on-the-fly (like HashHop's HashTokenizer).
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
import time
import sys
from pathlib import Path

# Import our CodeTokenizer
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import CodeTokenizer


class SimpleLTM(nn.Module):
    """Simple LTM model closer to HashHop architecture.

    Key-Value Memory:
    - Keys: function/class names (single tokens)
    - Values: function implementations (encoded sequences)

    Query mechanism:
    - Query attends over keys
    - Retrieves corresponding values
    - Decoder generates from retrieved context
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Shared embedding for all tokens
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Key projection (for memory lookup)
        self.key_proj = nn.Linear(d_model, d_model)

        # Value encoder (encode full implementations)
        self.value_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Query projection
        self.query_proj = nn.Linear(d_model, d_model)

        # Decoder layers (simple transformer)
        self.decoder_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]

        # Output
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def encode_memory(
        self,
        keys: mx.array,       # (num_items,) - single token per item (function name)
        values: mx.array,     # (num_items, val_len) - implementation tokens
    ) -> Tuple[mx.array, mx.array]:
        """Encode memory bank.

        Returns:
            key_emb: (num_items, d_model)
            val_emb: (num_items, d_model)
        """
        # Keys: just embed the single token
        key_emb = self.embed(keys)  # (num_items, d_model)
        key_emb = self.key_proj(key_emb)

        # Values: embed and mean pool
        val_emb = self.embed(values)  # (num_items, val_len, d_model)
        val_emb = mx.mean(val_emb, axis=1)  # (num_items, d_model)
        val_emb = self.value_encoder(val_emb)

        return key_emb, val_emb

    def retrieve(
        self,
        query_emb: mx.array,  # (batch, d_model) or (batch, seq, d_model)
        key_emb: mx.array,    # (num_items, d_model)
        val_emb: mx.array,    # (num_items, d_model)
        temperature: float = 1.0,
    ) -> Tuple[mx.array, mx.array]:
        """Retrieve from memory using attention.

        Returns:
            retrieved: (batch, d_model) or (batch, seq, d_model)
            attn_weights: attention weights
        """
        # Project query
        q = self.query_proj(query_emb)

        # Handle different query shapes
        if q.ndim == 2:
            # (batch, d_model) -> (batch, 1, d_model)
            q = q[:, None, :]
            squeeze = True
        else:
            squeeze = False

        # Attention: q @ k.T / sqrt(d)
        scale = self.d_model ** -0.5
        scores = (q @ key_emb.T) * scale / temperature  # (batch, seq, num_items)
        attn = mx.softmax(scores, axis=-1)

        # Retrieve: weighted sum of values
        retrieved = attn @ val_emb  # (batch, seq, d_model)

        if squeeze:
            retrieved = retrieved[:, 0, :]
            attn = attn[:, 0, :]

        return retrieved, attn

    def __call__(
        self,
        input_ids: mx.array,   # (batch, seq_len)
        key_emb: mx.array,     # (num_items, d_model)
        val_emb: mx.array,     # (num_items, d_model)
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass with memory retrieval.

        Returns:
            logits: (batch, seq_len, vocab_size)
            attn: retrieval attention weights
        """
        B, L = input_ids.shape

        # Embed input
        h = self.embed(input_ids)
        pos = mx.arange(L)
        h = h + self.pos_embed(pos)

        # Retrieve from memory based on full sequence
        # Use mean of sequence as query
        query = mx.mean(h, axis=1)  # (batch, d_model)
        retrieved, attn = self.retrieve(query, key_emb, val_emb)

        # Add retrieved context to all positions
        h = h + retrieved[:, None, :]  # Broadcast to all positions

        # Causal mask for decoder
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        # Apply decoder layers
        for layer in self.decoder_layers:
            h = layer(h, mask)

        # Output
        h = self.ln(h)
        logits = self.output(h)

        return logits, attn

    def generate(
        self,
        key_emb: mx.array,
        val_emb: mx.array,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
    ) -> Tuple[List[int], mx.array]:
        """Generate tokens with memory retrieval."""
        generated = list(prompt_ids)
        attn = None

        for _ in range(max_new_tokens):
            context = generated[-self.max_seq_len:]
            x = mx.array([context])

            logits, attn = self(x, key_emb, val_emb)
            logits = logits[0, -1] / temperature

            probs = mx.softmax(logits)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            generated.append(int(next_token))

            if next_token == 3:  # EOS
                break

        return generated, attn


class DynamicTokenizer:
    """Dynamic tokenizer that grows vocabulary as needed.

    Like HashHop's HashTokenizer - new symbols get new IDs automatically.
    No predefined vocabulary needed!
    """

    def __init__(self):
        # Start with minimal special tokens
        self.special = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.token_to_id = dict(self.special)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.next_id = len(self.special)

    def _add_token(self, token: str) -> int:
        """Add token to vocab if not present, return ID."""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]

    def encode(self, text: str) -> List[int]:
        """Tokenize text, adding new tokens to vocab as needed."""
        import re
        # Split on whitespace and operators, keeping identifiers whole
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=():\[\].,<>!@#$%^&|~`]|\d+\.?\d*|"[^"]*"|\'[^\']*\'|\S', text)
        return [self._add_token(t) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return " ".join(tokens)

    def vocab_size(self) -> int:
        return self.next_id

    def get_token_id(self, token: str) -> int:
        """Get ID for a token (adds if not present)."""
        return self._add_token(token)


def train_simple_ltm():
    """Train the simple LTM model."""
    print("=" * 60)
    print("Simple LTM Training (Dynamic Tokenizer)")
    print("=" * 60)

    # Create dynamic tokenizer - vocab grows as we see new code
    tokenizer = DynamicTokenizer()
    print("\nUsing dynamic tokenizer (vocab grows automatically)")

    # Create memory bank (function definitions) - ANY Python code works!
    memory_items = [
        ("add", "def add(a, b): return a + b"),
        ("subtract", "def subtract(a, b): return a - b"),
        ("multiply", "def multiply(a, b): return a * b"),
        ("divide", "def divide(a, b): return a / b"),
        ("square", "def square(x): return x * x"),
        ("double", "def double(x): return x + x"),
        ("factorial", "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)"),
        ("fibonacci", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
    ]

    # First pass: encode everything to build vocabulary
    print("\nBuilding vocabulary from code...")
    for name, impl in memory_items:
        tokenizer.encode(name)
        tokenizer.encode(impl)

    # Also encode some query patterns
    query_patterns = [
        "result = add(", "x = multiply(", "Call square", "Use divide",
        "calculate factorial", "compute fibonacci"
    ]
    for q in query_patterns:
        tokenizer.encode(q)

    print(f"Vocabulary size after indexing code: {tokenizer.vocab_size()}")

    # Now create model with correct vocab size
    model = SimpleLTM(
        vocab_size=tokenizer.vocab_size() + 100,  # Buffer for new tokens
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    import mlx.utils as mlx_utils
    flat_params = mlx_utils.tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    print(f"Model params: {num_params:,}")

    # Encode memory
    keys = mx.array([tokenizer.get_token_id(name) for name, _ in memory_items])
    max_val_len = 30
    values = []
    for _, impl in memory_items:
        ids = tokenizer.encode(impl)
        ids = ids[:max_val_len] + [0] * (max_val_len - len(ids))
        values.append(ids)
    values = mx.array(values)

    print(f"\nMemory: {len(memory_items)} items")
    print(f"  Keys shape: {keys.shape}")
    print(f"  Values shape: {values.shape}")

    # Encode memory once
    key_emb, val_emb = model.encode_memory(keys, values)
    print(f"  Key embeddings: {key_emb.shape}")
    print(f"  Value embeddings: {val_emb.shape}")

    # Training data: queries and expected function calls
    train_data = []
    for name, impl in memory_items:
        # Create query-target pairs with natural language variations
        queries = [
            f"result = {name}(a, b)",
            f"x = {name}(",
            f"Call {name}",
            f"Use {name} function",
            f"compute using {name}",
        ]
        for q in queries:
            input_ids = tokenizer.encode(q)
            # Target: continue with the function name
            target_ids = input_ids[1:] + [tokenizer.get_token_id(name)]
            train_data.append((input_ids, target_ids, name))

    print(f"\nTraining samples: {len(train_data)}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=1e-3)

    # Get actual vocab size for loss
    actual_vocab_size = model.vocab_size

    def loss_fn(model, input_ids, targets, key_emb, val_emb):
        logits, _ = model(input_ids, key_emb, val_emb)
        logits_flat = logits.reshape(-1, actual_vocab_size)
        targets_flat = targets.reshape(-1)
        return nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    num_steps = 500
    batch_size = 6
    log_every = 50
    eval_every = 100

    start_time = time.time()
    total_loss = 0

    for step in range(1, num_steps + 1):
        # Sample batch
        batch = random.sample(train_data, min(batch_size, len(train_data)))

        # Pad sequences
        max_len = max(len(x[0]) for x in batch)
        inputs = []
        targets = []
        for inp, tgt, _ in batch:
            inp = inp + [0] * (max_len - len(inp))
            tgt = tgt + [0] * (max_len - len(tgt))
            inputs.append(inp)
            targets.append(tgt)

        inputs = mx.array(inputs)
        targets = mx.array(targets)

        # Forward and backward
        loss, grads = loss_and_grad(model, inputs, targets, key_emb, val_emb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

        if step % log_every == 0:
            avg_loss = total_loss / log_every
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | loss: {avg_loss:.4f} | time: {elapsed:.1f}s")
            total_loss = 0

        if step % eval_every == 0:
            print("\n--- Evaluation ---")
            evaluate(model, key_emb, val_emb, tokenizer, memory_items)
            print("-" * 60)

    print("\nTraining complete!")
    return model, key_emb, val_emb, tokenizer


def evaluate(model, key_emb, val_emb, tokenizer, memory_items):
    """Evaluate retrieval accuracy."""
    # Build test queries dynamically from memory items
    test_queries = []
    for name, _ in memory_items[:6]:  # Test first 6 functions
        test_queries.append((f"result = {name}(", name))
        test_queries.append((f"Call {name}", name))

    correct = 0
    for query, expected in test_queries[:8]:  # Limit to 8 tests
        input_ids = tokenizer.encode(query)
        generated, attn = model.generate(
            key_emb, val_emb, input_ids,
            max_new_tokens=5, temperature=0.5
        )

        output = tokenizer.decode(generated[len(input_ids):])

        # Check what memory slot got highest attention
        if attn is not None:
            top_idx = int(mx.argmax(attn[0]))
            if top_idx < len(memory_items):
                retrieved_name = memory_items[top_idx][0]
            else:
                retrieved_name = "?"
        else:
            retrieved_name = "?"

        is_correct = expected in output or retrieved_name == expected
        if is_correct:
            correct += 1

        print(f"  Q: '{query}' -> Retrieved: {retrieved_name} | Output: {output[:15]}")
        print(f"    Expected: {expected} {'✓' if is_correct else '✗'}")

    print(f"\n  Accuracy: {correct}/{len(test_queries[:8])}")


if __name__ == "__main__":
    train_simple_ltm()
