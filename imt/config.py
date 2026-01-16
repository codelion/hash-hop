"""Configuration dataclasses for Indexed Memory Transformer."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IMTConfig:
    """Configuration for Indexed Memory Transformer model.

    Architecture designed for HashHop retrieval tasks on Apple Silicon using MLX.
    Default values are for IMT-Nano (~12-15M parameters).
    """

    # Encoder configuration
    d_model: int = 256
    encoder_layers: int = 2
    encoder_heads: int = 4
    encoder_ff_dim: int = 1024
    chunk_size: int = 512  # tokens per chunk

    # Index configuration
    index_dim: int = 128
    num_index_heads: int = 4
    keys_per_chunk: int = 10
    num_clusters: int = 256
    temperature: float = 1.0  # softmax temperature for soft retrieval

    # Decoder configuration
    decoder_layers: int = 3
    decoder_heads: int = 4
    decoder_ff_dim: int = 1024
    retrieval_top_k: int = 4

    # Tokenization
    vocab_size: int = 70  # Will be set by tokenizer
    max_hash_length: int = 20  # Max length of hash string + padding

    # Training
    dropout: float = 0.1
    max_chunks: int = 20000  # for 10M context

    # Memory optimization
    use_gradient_checkpointing: bool = True
    chunk_batch_size: int = 64  # chunks processed per batch during encoding
    memory_limit_fraction: float = 0.5  # Limit VRAM usage to this fraction (0.0-1.0)

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.d_model % self.encoder_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by encoder_heads ({self.encoder_heads})"
        )
        assert self.d_model % self.decoder_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by decoder_heads ({self.decoder_heads})"
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters for IMT."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation: int = 4

    # Evaluation and checkpointing
    eval_every: int = 500
    eval_samples: int = 10
    save_every: int = 2000
    log_every: int = 10

    # Data generation parameters
    n_chars_problem: int = 10_000_000  # 10M characters (~10M tokens for char-level)
    num_queries: int = 10  # queries per sample
    hops: int = 2  # number of hops for multi-hop retrieval
    hash_pair_str_length: int = 16

    # Loss weights
    lambda_retrieval: float = 0.5
    lambda_regularization: float = 0.01
    lambda_contrastive: float = 1.0  # Weight for contrastive embedding loss
    lambda_copy: float = 0.1  # Weight for copy gate loss
    lambda_copy_attn: float = 1.0  # Weight for copy attention supervision loss

    # Output
    output_dir: str = "checkpoints"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Set default experiment name if not provided."""
        if self.experiment_name is None:
            import time
            self.experiment_name = f"imt_{int(time.time())}"
