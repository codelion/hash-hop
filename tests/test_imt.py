"""Tests for Indexed Memory Transformer components."""

import pytest

import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.data.tokenizer import HashTokenizer
from imt.model.chunk_encoder import ChunkEncoder
from imt.model.decoder import LocalDecoder
from imt.model.index import LearnedIndexSearch
from imt.model.imt import IndexedMemoryTransformer
from imt.model.key_extractor import IndexKeyExtractor
from imt.training.loss import (
    compute_accuracy,
    compute_generation_loss,
    compute_retrieval_loss,
    compute_retrieval_recall,
)


@pytest.fixture
def small_config() -> IMTConfig:
    """Create a small config for testing."""
    return IMTConfig(
        d_model=64,
        encoder_layers=1,
        encoder_heads=2,
        encoder_ff_dim=128,
        chunk_size=32,
        index_dim=32,
        num_index_heads=2,
        keys_per_chunk=4,
        num_clusters=16,
        retrieval_top_k=2,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ff_dim=128,
        vocab_size=70,
        max_hash_length=16,
        dropout=0.0,
        chunk_batch_size=8,
    )


@pytest.fixture
def tokenizer() -> HashTokenizer:
    """Create tokenizer fixture."""
    return HashTokenizer()


class TestHashTokenizer:
    """Tests for HashTokenizer."""

    def test_vocab_size(self, tokenizer: HashTokenizer) -> None:
        """Test vocab size is correct."""
        # 9 special + 26 lowercase + 26 uppercase + 10 digits = 71
        assert tokenizer.vocab_size == 71

    def test_encode_decode_roundtrip(self, tokenizer: HashTokenizer) -> None:
        """Test encode/decode roundtrip."""
        text = "abcXYZ123"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_encode_special_tokens(self, tokenizer: HashTokenizer) -> None:
        """Test encoding special tokens."""
        text = "a = 'b'\n"
        encoded = tokenizer.encode(text)
        assert len(encoded) == len(text)
        # Check special tokens are encoded correctly
        assert tokenizer.char_to_id["="] in encoded
        assert tokenizer.char_to_id["'"] in encoded
        assert tokenizer.char_to_id["\n"] in encoded

    def test_encode_batch(self, tokenizer: HashTokenizer) -> None:
        """Test batch encoding with padding."""
        texts = ["abc", "defgh", "x"]
        encoded = tokenizer.encode_batch(texts, max_length=8)
        assert encoded.shape == (3, 8)
        # Check padding
        assert encoded[2, 1].item() == tokenizer.pad_id


class TestChunkEncoder:
    """Tests for ChunkEncoder."""

    def test_output_shape(self, small_config: IMTConfig) -> None:
        """Test encoder output shape."""
        encoder = ChunkEncoder(small_config)
        batch_size = 4
        tokens = mx.zeros((batch_size, small_config.chunk_size), dtype=mx.int32)

        output = encoder(tokens)

        assert output.shape == (batch_size, small_config.chunk_size, small_config.d_model)

    def test_batched_encoding(self, small_config: IMTConfig) -> None:
        """Test batched encoding for multiple chunks."""
        encoder = ChunkEncoder(small_config)
        num_chunks = 20
        tokens = mx.zeros((num_chunks, small_config.chunk_size), dtype=mx.int32)

        output = encoder.encode_batched(tokens, batch_size=8)

        assert output.shape == (num_chunks, small_config.chunk_size, small_config.d_model)


class TestIndexKeyExtractor:
    """Tests for IndexKeyExtractor."""

    def test_output_shapes(self, small_config: IMTConfig) -> None:
        """Test key extractor output shapes."""
        extractor = IndexKeyExtractor(small_config)
        batch_size = 4
        hidden = mx.zeros((batch_size, small_config.chunk_size, small_config.d_model))

        keys, values, weights = extractor(hidden)

        assert keys.shape == (batch_size, small_config.keys_per_chunk, small_config.index_dim)
        assert values.shape == (batch_size, small_config.keys_per_chunk, small_config.index_dim)
        assert weights.shape == (batch_size, small_config.keys_per_chunk, small_config.chunk_size)

    def test_attention_weights_sum_to_one(self, small_config: IMTConfig) -> None:
        """Test that attention weights sum to 1."""
        extractor = IndexKeyExtractor(small_config)
        hidden = mx.random.normal((2, small_config.chunk_size, small_config.d_model))

        _, _, weights = extractor(hidden)

        # Attention weights should sum to 1 along the sequence dimension
        sums = mx.sum(weights, axis=-1)
        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5)


class TestLearnedIndexSearch:
    """Tests for LearnedIndexSearch."""

    def test_build_index(self, small_config: IMTConfig) -> None:
        """Test index building."""
        index_search = LearnedIndexSearch(small_config)

        num_chunks = 50
        keys = mx.random.normal((num_chunks, small_config.keys_per_chunk, small_config.index_dim))
        values = mx.random.normal((num_chunks, small_config.keys_per_chunk, small_config.index_dim))
        chunk_hidden = mx.random.normal(
            (num_chunks, small_config.chunk_size, small_config.d_model)
        )

        index = index_search.build_index(keys, values, chunk_hidden)

        assert "keys" in index
        assert "flat_keys" in index
        assert "cluster_assignments" in index
        assert index["num_chunks"] == num_chunks

    def test_search(self, small_config: IMTConfig) -> None:
        """Test index search."""
        index_search = LearnedIndexSearch(small_config)

        num_chunks = 50
        keys = mx.random.normal((num_chunks, small_config.keys_per_chunk, small_config.index_dim))
        values = mx.random.normal((num_chunks, small_config.keys_per_chunk, small_config.index_dim))
        chunk_hidden = mx.random.normal(
            (num_chunks, small_config.chunk_size, small_config.d_model)
        )

        index = index_search.build_index(keys, values, chunk_hidden)

        batch_size = 8
        query = mx.random.normal((batch_size, small_config.d_model))

        retrieved, scores, indices = index_search.search(query, index)

        assert retrieved.shape == (
            batch_size,
            small_config.retrieval_top_k,
            small_config.chunk_size,
            small_config.d_model,
        )
        assert scores.shape == (batch_size, small_config.retrieval_top_k)
        assert indices.shape == (batch_size, small_config.retrieval_top_k)

    def test_indices_in_range(self, small_config: IMTConfig) -> None:
        """Test that retrieved indices are valid."""
        index_search = LearnedIndexSearch(small_config)

        num_chunks = 50
        keys = mx.random.normal((num_chunks, small_config.keys_per_chunk, small_config.index_dim))
        values = mx.random.normal((num_chunks, small_config.keys_per_chunk, small_config.index_dim))
        chunk_hidden = mx.random.normal(
            (num_chunks, small_config.chunk_size, small_config.d_model)
        )

        index = index_search.build_index(keys, values, chunk_hidden)
        query = mx.random.normal((4, small_config.d_model))

        _, _, indices = index_search.search(query, index)

        # All indices should be in valid range
        assert mx.all(indices >= 0)
        assert mx.all(indices < num_chunks)


class TestLocalDecoder:
    """Tests for LocalDecoder."""

    def test_output_shape(self, small_config: IMTConfig) -> None:
        """Test decoder output shape."""
        decoder = LocalDecoder(small_config)

        batch_size = 4
        query_len = small_config.max_hash_length
        query_tokens = mx.zeros((batch_size, query_len), dtype=mx.int32)
        retrieved_chunks = mx.random.normal(
            (batch_size, small_config.retrieval_top_k, small_config.chunk_size, small_config.d_model)
        )

        logits, query_repr = decoder(query_tokens, retrieved_chunks)

        assert logits.shape == (batch_size, query_len, small_config.vocab_size)
        assert query_repr.shape == (batch_size, small_config.d_model)


class TestIndexedMemoryTransformer:
    """Tests for full IMT model."""

    def test_forward_pass(self, small_config: IMTConfig) -> None:
        """Test full forward pass."""
        model = IndexedMemoryTransformer(small_config)

        num_chunks = 20
        chunk_tokens = mx.zeros((num_chunks, small_config.chunk_size), dtype=mx.int32)
        query_tokens = mx.zeros((4, small_config.max_hash_length), dtype=mx.int32)

        logits, scores, indices, keys = model(chunk_tokens, query_tokens)

        assert logits.shape == (4, small_config.max_hash_length, small_config.vocab_size)
        assert scores.shape == (4, small_config.retrieval_top_k)
        assert indices.shape == (4, small_config.retrieval_top_k)
        assert keys.shape == (num_chunks, small_config.keys_per_chunk, small_config.index_dim)

    def test_precomputed_index(self, small_config: IMTConfig) -> None:
        """Test using precomputed index."""
        model = IndexedMemoryTransformer(small_config)

        num_chunks = 20
        chunk_tokens = mx.zeros((num_chunks, small_config.chunk_size), dtype=mx.int32)
        query_tokens = mx.zeros((4, small_config.max_hash_length), dtype=mx.int32)

        # First pass to build index
        _, _, _, _ = model(chunk_tokens, query_tokens)

        # Build index separately
        chunk_hidden, index_keys, index_values = model.encode_chunks(chunk_tokens)
        precomputed_index = model.build_index(chunk_hidden, index_keys, index_values)

        # Use precomputed index
        logits, scores, indices, _ = model(
            chunk_tokens, query_tokens, precomputed_index=precomputed_index
        )

        assert logits.shape == (4, small_config.max_hash_length, small_config.vocab_size)

    def test_parameter_count(self, small_config: IMTConfig) -> None:
        """Test parameter counting."""
        model = IndexedMemoryTransformer(small_config)
        num_params = model.count_parameters()

        # Should have some parameters
        assert num_params > 0
        # For small config, should be reasonably small
        assert num_params < 1_000_000


class TestLossFunctions:
    """Tests for loss functions."""

    def test_generation_loss(self) -> None:
        """Test generation loss computation."""
        batch_size, seq_len, vocab_size = 4, 10, 70

        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_generation_loss(logits, targets, pad_id=0)

        assert loss.shape == ()  # Scalar
        assert float(loss) > 0  # Should be positive

    def test_retrieval_loss(self) -> None:
        """Test retrieval loss computation."""
        batch_size, top_k = 4, 4

        scores = mx.softmax(mx.random.normal((batch_size, top_k)), axis=-1)
        chunk_indices = mx.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        target_indices = mx.array([0, 1, 10, 3])  # Some in retrieved, some not

        loss = compute_retrieval_loss(scores, chunk_indices, target_indices)

        assert loss.shape == ()  # Scalar

    def test_accuracy(self) -> None:
        """Test accuracy computation."""
        batch_size, seq_len, vocab_size = 4, 10, 70

        # Create targets
        targets = mx.random.randint(1, vocab_size, (batch_size, seq_len))  # Avoid 0 (pad)

        # Create logits with one-hot encoding (high value at target position)
        # Use scatter-like approach via creating mask
        logits = mx.zeros((batch_size, seq_len, vocab_size))
        targets_expanded = targets[:, :, None]  # (batch, seq, 1)
        indices = mx.arange(vocab_size)[None, None, :]  # (1, 1, vocab_size)
        mask = (indices == targets_expanded).astype(mx.float32)  # (batch, seq, vocab_size)
        logits = logits + mask * 10.0

        accuracy = compute_accuracy(logits, targets, pad_id=0)

        assert 0 <= accuracy <= 1
        assert accuracy > 0.9  # Should be high since we set it up to match

    def test_retrieval_recall(self) -> None:
        """Test retrieval recall computation."""
        chunk_indices = mx.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        target_indices = mx.array([0, 1, 10, 3])  # 3 out of 4 are in retrieved

        recall = compute_retrieval_recall(chunk_indices, target_indices)

        assert recall == 0.75  # 3/4


class TestIntegration:
    """Integration tests."""

    def test_training_step_runs(self, small_config: IMTConfig) -> None:
        """Test that a training step can run without errors."""
        from imt.data.chunked_dataset import ChunkedHashHopDataset
        from imt.training.loss import compute_total_loss

        # Create small training config
        train_config = TrainingConfig(
            n_chars_problem=10000,  # Very small for testing
            num_queries=2,
            hops=1,
            hash_pair_str_length=8,
        )

        tokenizer = HashTokenizer()
        small_config.vocab_size = tokenizer.vocab_size

        model = IndexedMemoryTransformer(small_config)
        dataset = ChunkedHashHopDataset(small_config, train_config, tokenizer)

        # Generate sample
        sample = dataset.generate_sample()

        # Forward pass
        logits, scores, indices, keys = model(sample.chunk_tokens, sample.query_tokens)

        # Compute loss
        loss, metrics = compute_total_loss(
            logits=logits,
            targets=sample.target_tokens,
            retrieval_scores=scores,
            chunk_indices=indices,
            target_chunk_indices=sample.target_chunk_indices,
            index_keys=keys,
            centroids=model.index_search.centroids,
            pad_id=tokenizer.pad_id,
        )

        assert loss.shape == ()
        assert "total_loss" in metrics
        assert "generation_loss" in metrics
        assert "retrieval_loss" in metrics
