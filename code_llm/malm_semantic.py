"""MALM with Semantic Query Support.

This extends MALM v2 to support natural language queries like:
- "find the function that adds two numbers"
- "function for user authentication"
- "how to calculate the sum"

Key addition: Train a query encoder that maps NL descriptions to
the function embedding space using contrastive learning.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
from typing import List, Dict, Tuple
import random
import time
import re
import ast


class SemanticMALM(nn.Module):
    """MALM with semantic query support.

    Architecture:
    - Function name embeddings (single tokens, like MALM v2)
    - Query encoder (maps variable-length NL to single embedding)
    - Contrastive training aligns queries with function embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Query encoder - transforms variable-length query to single embedding
        self.query_encoder_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(2)  # Smaller encoder for queries
        ]
        self.query_proj = nn.Linear(d_model, d_model)

        # Value encoder - encodes function implementations
        self.value_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Decoder for generation
        self.decoder_layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]

        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def encode_query(self, query_ids: mx.array) -> mx.array:
        """Encode a variable-length query to a single embedding.

        Args:
            query_ids: (batch, seq_len) token IDs

        Returns:
            query_emb: (batch, d_model) single embedding per query
        """
        B, L = query_ids.shape

        # Embed
        h = self.embed(query_ids)
        pos = mx.arange(L)
        h = h + self.pos_embed(pos)

        # Self-attention
        for layer in self.query_encoder_layers:
            h = layer(h, None)

        # Mean pool to get single embedding
        mask = (query_ids != 0).astype(mx.float32)[:, :, None]
        h = h * mask
        query_emb = mx.sum(h, axis=1) / (mx.sum(mask, axis=1) + 1e-8)

        return self.query_proj(query_emb)

    def encode_memory(
        self,
        key_tokens: mx.array,    # (num_items,) function name tokens
        value_tokens: mx.array,  # (num_items, val_len) implementation tokens
    ) -> Tuple[mx.array, mx.array]:
        """Encode memory bank."""
        # Keys: function name embeddings
        key_emb = self.embed(key_tokens)

        # Values: encoded implementations
        val_emb = self.embed(value_tokens)
        mask = (value_tokens != 0).astype(mx.float32)[:, :, None]
        val_emb = val_emb * mask
        val_emb = mx.sum(val_emb, axis=1) / (mx.sum(mask, axis=1) + 1e-8)
        val_emb = self.value_encoder(val_emb)

        return key_emb, val_emb

    def retrieve(
        self,
        query_emb: mx.array,  # (batch, d_model)
        key_emb: mx.array,    # (num_items, d_model)
        val_emb: mx.array,    # (num_items, d_model)
        temperature: float = 1.0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Retrieve from memory using query embedding."""
        scale = self.d_model ** -0.5
        scores = (query_emb @ key_emb.T) * scale / temperature
        attn = mx.softmax(scores, axis=-1)
        retrieved = attn @ val_emb
        return retrieved, attn, scores

    def forward(
        self,
        query_ids: mx.array,     # (batch, query_len) - NL query tokens
        key_emb: mx.array,
        val_emb: mx.array,
        continuation: mx.array,  # (batch, cont_len) - tokens to generate
        temperature: float = 1.0,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward with semantic query."""
        B, L = continuation.shape

        # Encode query
        query_emb = self.encode_query(query_ids)

        # Retrieve
        retrieved, attn, scores = self.retrieve(query_emb, key_emb, val_emb, temperature)

        # Embed continuation
        h = self.embed(continuation)
        pos = mx.arange(L)
        h = h + self.pos_embed(pos)

        # Add retrieved context
        h = h + retrieved[:, None, :]

        # Decode
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        for layer in self.decoder_layers:
            h = layer(h, mask)

        h = self.ln(h)
        logits = self.output(h)

        return logits, attn, scores

    def __call__(self, query_ids, key_emb, val_emb, continuation, temperature=1.0):
        return self.forward(query_ids, key_emb, val_emb, continuation, temperature)


class SemanticTokenizer:
    """Tokenizer that handles both code and natural language."""

    def __init__(self):
        self.special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.token_to_id = dict(self.special)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.next_id = len(self.special)

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]

    def encode(self, text: str) -> List[int]:
        # Split on whitespace and common delimiters
        tokens = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=():\[\].,<>!@#$%^&|~`]|\d+\.?\d*|"[^"]*"|\'[^\']*\'|\S',
            text.lower()  # Lowercase for better matching
        )
        return [self.add_token(t) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<UNK>") for i in ids)

    def vocab_size(self) -> int:
        return self.next_id


def generate_query_variations(name: str, docstring: str, source: str) -> List[str]:
    """Generate natural language query variations for a function."""
    queries = []

    # Name-based (these are easy)
    queries.append(f"function {name}")
    queries.append(f"find {name}")
    queries.append(f"get {name}")

    # Docstring-based (if available)
    if docstring:
        words = docstring.lower().split()[:6]
        if len(words) >= 2:
            queries.append(" ".join(words))
            queries.append("function that " + " ".join(words[:4]))
            queries.append("find function for " + " ".join(words[:3]))

    # Name decomposition (e.g., "calculate_sum" -> "calculate sum")
    name_parts = name.replace("_", " ").lower()
    if name_parts != name.lower():
        queries.append(name_parts)
        queries.append(f"function to {name_parts}")

    # Common patterns
    if "add" in name.lower() or "sum" in name.lower():
        queries.extend(["add numbers", "sum values", "addition function"])
    if "auth" in name.lower() or "login" in name.lower():
        queries.extend(["user authentication", "login function", "check credentials"])
    if "multiply" in name.lower() or "product" in name.lower():
        queries.extend(["multiply numbers", "product function", "multiplication"])
    if "subtract" in name.lower() or "diff" in name.lower():
        queries.extend(["subtract numbers", "difference", "subtraction"])
    if "divide" in name.lower() or "div" in name.lower():
        queries.extend(["divide numbers", "division", "quotient"])
    if "sort" in name.lower():
        queries.extend(["sort list", "sorting function", "order elements"])
    if "search" in name.lower() or "find" in name.lower():
        queries.extend(["search function", "find element", "lookup"])
    if "parse" in name.lower():
        queries.extend(["parse data", "parsing function", "extract information"])
    if "format" in name.lower():
        queries.extend(["format data", "formatting", "convert to string"])
    if "validate" in name.lower() or "check" in name.lower():
        queries.extend(["validate input", "check data", "verification"])

    return list(set(queries))  # Remove duplicates


def train_semantic_malm(
    max_memory: int = 100,
    num_steps: int = 3000,
    batch_size: int = 16,
    d_model: int = 256,
    lr: float = 1e-3,
):
    """Train MALM with semantic query support."""

    print("=" * 70)
    print("Semantic MALM: Natural Language Queries for Code")
    print("=" * 70)

    tokenizer = SemanticTokenizer()

    # Sample functions with descriptions
    memory_items = [
        {"name": "add", "source": "def add(a, b): return a + b", "docstring": "Add two numbers together"},
        {"name": "subtract", "source": "def subtract(a, b): return a - b", "docstring": "Subtract b from a"},
        {"name": "multiply", "source": "def multiply(a, b): return a * b", "docstring": "Multiply two numbers"},
        {"name": "divide", "source": "def divide(a, b): return a / b", "docstring": "Divide a by b"},
        {"name": "square", "source": "def square(x): return x * x", "docstring": "Calculate the square of x"},
        {"name": "authenticate", "source": "def authenticate(user, pwd): return verify_password(user, pwd)", "docstring": "Authenticate user credentials"},
        {"name": "login", "source": "def login(username, password): return create_session(username)", "docstring": "Login user and create session"},
        {"name": "logout", "source": "def logout(session_id): return end_session(session_id)", "docstring": "Logout user and end session"},
        {"name": "sort_list", "source": "def sort_list(items): return sorted(items)", "docstring": "Sort a list of items"},
        {"name": "find_max", "source": "def find_max(items): return max(items)", "docstring": "Find maximum value in list"},
        {"name": "find_min", "source": "def find_min(items): return min(items)", "docstring": "Find minimum value in list"},
        {"name": "calculate_average", "source": "def calculate_average(nums): return sum(nums) / len(nums)", "docstring": "Calculate average of numbers"},
        {"name": "parse_json", "source": "def parse_json(text): return json.loads(text)", "docstring": "Parse JSON string to object"},
        {"name": "format_date", "source": "def format_date(dt): return dt.strftime('%Y-%m-%d')", "docstring": "Format date to string"},
        {"name": "validate_email", "source": "def validate_email(email): return '@' in email", "docstring": "Validate email address format"},
        {"name": "hash_password", "source": "def hash_password(pwd): return hashlib.sha256(pwd.encode()).hexdigest()", "docstring": "Hash password securely"},
        {"name": "generate_token", "source": "def generate_token(): return secrets.token_hex(32)", "docstring": "Generate secure random token"},
        {"name": "fetch_data", "source": "def fetch_data(url): return requests.get(url).json()", "docstring": "Fetch data from URL"},
        {"name": "save_file", "source": "def save_file(path, content): open(path, 'w').write(content)", "docstring": "Save content to file"},
        {"name": "read_file", "source": "def read_file(path): return open(path).read()", "docstring": "Read content from file"},
    ]

    if max_memory < len(memory_items):
        memory_items = memory_items[:max_memory]

    print(f"\nMemory items: {len(memory_items)} functions")

    # Build vocabulary and generate queries
    all_queries = []  # (query_text, target_idx)

    for idx, item in enumerate(memory_items):
        # Add tokens
        tokenizer.add_token(item["name"])
        tokenizer.encode(item["source"])
        tokenizer.encode(item.get("docstring", ""))

        # Generate query variations
        queries = generate_query_variations(
            item["name"],
            item.get("docstring", ""),
            item["source"]
        )
        for q in queries:
            tokenizer.encode(q)
            all_queries.append((q, idx))

    print(f"Query variations: {len(all_queries)}")
    print(f"Vocab size: {tokenizer.vocab_size()}")

    # Build memory arrays
    keys = [tokenizer.add_token(item["name"]) for item in memory_items]
    values = []
    max_val_len = 40

    for item in memory_items:
        ids = tokenizer.encode(item["source"])
        ids = ids[:max_val_len] + [0] * (max_val_len - len(ids))
        values.append(ids)

    keys = mx.array(keys)
    values = mx.array(values)

    # Create model
    model = SemanticMALM(
        vocab_size=tokenizer.vocab_size() + 100,
        d_model=d_model,
        n_heads=8,
        n_layers=3,
    )

    flat_params = mlx_utils.tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    print(f"Model parameters: {num_params:,}")

    # Encode memory
    key_emb, val_emb = model.encode_memory(keys, values)
    print(f"Memory encoded: {key_emb.shape}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, query_ids, target_idx, key_emb, val_emb, temp):
        cont = mx.zeros((query_ids.shape[0], 1), dtype=mx.int32) + 2  # BOS
        _, _, scores = model(query_ids, key_emb, val_emb, cont, temp)

        # Contrastive loss
        scores_scaled = scores / 0.07  # Temperature for contrastive
        return nn.losses.cross_entropy(scores_scaled, target_idx, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 60)

    start_time = time.time()
    total_loss = 0.0
    max_query_len = 15

    for step in range(1, num_steps + 1):
        temp = max(0.5, 1.5 - step / num_steps)

        # Sample batch of queries
        batch = random.sample(all_queries, min(batch_size, len(all_queries)))

        # Pad queries
        query_ids = []
        target_idx = []
        for q, idx in batch:
            ids = tokenizer.encode(q)
            ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
            query_ids.append(ids)
            target_idx.append(idx)

        query_ids = mx.array(query_ids)
        target_idx = mx.array(target_idx)

        loss, grads = loss_and_grad(model, query_ids, target_idx, key_emb, val_emb, temp)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

        if step % 200 == 0:
            avg_loss = total_loss / 200
            elapsed = time.time() - start_time

            # Check accuracy
            cont = mx.zeros((batch_size, 1), dtype=mx.int32) + 2
            _, attn, _ = model(query_ids, key_emb, val_emb, cont, temp)
            pred_idx = mx.argmax(attn, axis=1)
            acc = mx.mean((pred_idx == target_idx).astype(mx.float32)).item()

            print(f"Step {step:4d} | loss={avg_loss:.4f} | acc={acc:.0%} | temp={temp:.2f} | time={elapsed:.0f}s")
            total_loss = 0.0

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")

    return model, tokenizer, key_emb, val_emb, memory_items, all_queries


def evaluate_semantic_malm(model, tokenizer, key_emb, val_emb, memory_items):
    """Evaluate semantic query capabilities."""

    print("\n" + "=" * 70)
    print("EVALUATION: Semantic Queries")
    print("=" * 70)

    # Test with semantic queries (NOT function names)
    test_queries = [
        ("add two numbers", "add"),
        ("sum values", "add"),
        ("subtract numbers", "subtract"),
        ("multiply values", "multiply"),
        ("division function", "divide"),
        ("user authentication", "authenticate"),
        ("login function", "login"),
        ("end user session", "logout"),
        ("sort a list", "sort_list"),
        ("find maximum value", "find_max"),
        ("calculate average", "calculate_average"),
        ("parse json data", "parse_json"),
        ("format date string", "format_date"),
        ("validate email address", "validate_email"),
        ("secure password hash", "hash_password"),
        ("fetch from url", "fetch_data"),
        ("save to file", "save_file"),
        ("read from file", "read_file"),
    ]

    # Filter to only include queries for functions in memory
    func_names = {item["name"] for item in memory_items}
    test_queries = [(q, f) for q, f in test_queries if f in func_names]

    max_query_len = 15
    correct = 0

    print("\nSemantic Query Results:")
    print("-" * 60)

    for query_text, expected_func in test_queries:
        # Encode query
        ids = tokenizer.encode(query_text)
        ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
        query_ids = mx.array([ids])

        cont = mx.array([[2]])  # BOS
        _, attn, _ = model(query_ids, key_emb, val_emb, cont, temperature=0.5)

        pred_idx = int(mx.argmax(attn[0]))
        predicted_func = memory_items[pred_idx]["name"]

        expected_idx = next(i for i, item in enumerate(memory_items) if item["name"] == expected_func)
        is_correct = pred_idx == expected_idx

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  '{query_text:25s}' -> '{predicted_func:20s}' (expected: {expected_func:15s}) {status}")

    print(f"\nSemantic Query Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.0f}%)")

    # Also test exact function names (should still work)
    print("\n" + "-" * 60)
    print("Exact Name Queries (baseline):")
    print("-" * 60)

    exact_correct = 0
    for item in memory_items[:10]:
        ids = tokenizer.encode(f"function {item['name']}")
        ids = ids[:max_query_len] + [0] * (max_query_len - len(ids))
        query_ids = mx.array([ids])

        cont = mx.array([[2]])
        _, attn, _ = model(query_ids, key_emb, val_emb, cont, temperature=0.5)

        pred_idx = int(mx.argmax(attn[0]))
        expected_idx = memory_items.index(item)

        if pred_idx == expected_idx:
            exact_correct += 1

        print(f"  'function {item['name']}' -> idx {pred_idx} (expected {expected_idx}) {'✓' if pred_idx == expected_idx else '✗'}")

    print(f"\nExact Name Accuracy: {exact_correct}/10 ({100*exact_correct/10:.0f}%)")


if __name__ == "__main__":
    model, tokenizer, key_emb, val_emb, memory_items, all_queries = train_semantic_malm(
        max_memory=20,
        num_steps=3000,
        batch_size=16,
        d_model=256,
        lr=1e-3,
    )

    evaluate_semantic_malm(model, tokenizer, key_emb, val_emb, memory_items)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Semantic MALM extends the base MALM to support natural language queries.

Key additions:
1. Query encoder - maps variable-length NL to single embedding
2. Contrastive training - aligns NL queries with function embeddings
3. Query variations - generates training data from function metadata

Limitations:
- Requires training data (query, function) pairs
- Limited to vocabulary seen during training
- Code editing still not supported (needs encoder-decoder)

For production use, consider:
- Using a pretrained code LLM as the base
- Adding more query variations automatically
- Fine-tuning on real user queries
""")
