"""Loss functions for IMT training."""

from typing import Dict, Tuple

import mlx.core as mx


def compute_generation_loss(
    logits: mx.array,
    targets: mx.array,
    pad_id: int = 0,
) -> mx.array:
    """Compute cross-entropy loss for answer generation.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        pad_id: Padding token ID to ignore in loss.

    Returns:
        Scalar loss value.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute log softmax
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

    # Gather log probs for target tokens
    # Create indices for gathering
    batch_indices = mx.arange(logits_flat.shape[0])
    target_log_probs = log_probs[batch_indices, targets_flat]

    # Mask padding tokens
    mask = (targets_flat != pad_id).astype(mx.float32)
    loss = -target_log_probs * mask

    return loss.sum() / (mask.sum() + 1e-8)


def compute_retrieval_loss(
    retrieval_scores: mx.array,
    chunk_indices: mx.array,
    target_chunk_indices: mx.array,
    all_chunk_scores: mx.array = None,
) -> mx.array:
    """Loss to encourage retrieving correct chunks.

    Uses cross-entropy over ALL chunk scores when available (preferred),
    otherwise falls back to margin loss over retrieved chunks.

    Args:
        retrieval_scores: Scores of shape (batch, top_k).
        chunk_indices: Retrieved chunk indices of shape (batch, top_k).
        target_chunk_indices: Ground truth chunk indices of shape (batch,).
        all_chunk_scores: Optional scores for ALL chunks of shape (batch, num_chunks).
            When provided, uses cross-entropy loss for direct supervision.

    Returns:
        Scalar loss value.
    """
    if all_chunk_scores is not None:
        # PREFERRED: Cross-entropy loss over all chunks
        # This provides gradient signal even when target isn't in top-k
        batch_size, num_chunks = all_chunk_scores.shape

        # Softmax over all chunks to get probabilities
        log_probs = mx.log(mx.softmax(all_chunk_scores, axis=-1) + 1e-10)

        # Gather log prob for target chunk
        batch_indices = mx.arange(batch_size)
        target_log_probs = log_probs[batch_indices, target_chunk_indices]

        return -mx.mean(target_log_probs)

    # Fallback: margin loss over retrieved chunks only
    target_expanded = target_chunk_indices[:, None]  # (batch, 1)
    is_target = (chunk_indices == target_expanded).astype(mx.float32)  # (batch, top_k)

    # Compute scores for target vs non-target
    target_scores = (retrieval_scores * is_target).sum(axis=-1)
    non_target_scores = (retrieval_scores * (1 - is_target)).sum(axis=-1)

    # Margin loss: target should score higher than non-targets
    margin = 0.1
    loss = mx.maximum(mx.array(0.0), non_target_scores - target_scores + margin)

    return mx.mean(loss)


def compute_index_regularization(
    index_keys: mx.array,
    centroids: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    """Regularization to encourage diverse, well-separated keys and clusters.

    Three objectives:
    1. Keys should be close to their assigned centroid
    2. Centroids should be spread apart (diverse)
    3. Chunk-level key representations should be diverse (NEW)

    Args:
        index_keys: Keys of shape (num_chunks, keys_per_chunk, index_dim).
        centroids: Cluster centroids of shape (num_clusters, index_dim).
        temperature: Softmax temperature for cluster assignment.

    Returns:
        Scalar regularization loss.
    """
    num_chunks = index_keys.shape[0]

    # Flatten keys
    flat_keys = index_keys.reshape(-1, index_keys.shape[-1])

    # Normalize for cosine similarity
    key_norms = mx.linalg.norm(flat_keys, axis=-1, keepdims=True)
    centroid_norms = mx.linalg.norm(centroids, axis=-1, keepdims=True)

    normalized_keys = flat_keys / (key_norms + 1e-8)
    normalized_centroids = centroids / (centroid_norms + 1e-8)

    # Compute key-centroid similarities
    similarities = mx.matmul(normalized_keys, normalized_centroids.T)

    # Loss 1: Keys should be close to their nearest centroid
    max_sim = mx.max(similarities, axis=-1)
    key_centroid_loss = -mx.mean(max_sim)

    # Loss 2: Centroids should be diverse (minimize pairwise similarity)
    centroid_sim = mx.matmul(normalized_centroids, normalized_centroids.T)
    # Mask diagonal (self-similarity)
    num_centroids = centroids.shape[0]
    mask = 1 - mx.eye(num_centroids)
    centroid_diversity_loss = mx.mean(centroid_sim * mask)

    # Loss 3: Chunk keys should be diverse (encourage different chunks to have different keys)
    # Average keys per chunk to get chunk-level representation
    chunk_key_repr = mx.mean(index_keys, axis=1)  # (num_chunks, index_dim)
    chunk_key_norm = chunk_key_repr / (mx.linalg.norm(chunk_key_repr, axis=-1, keepdims=True) + 1e-8)
    chunk_sim = mx.matmul(chunk_key_norm, chunk_key_norm.T)  # (num_chunks, num_chunks)

    # Penalize high similarity between different chunks
    chunk_mask = 1 - mx.eye(num_chunks)
    chunk_diversity_loss = mx.mean(chunk_sim * chunk_mask)

    return key_centroid_loss + 0.1 * centroid_diversity_loss + 0.5 * chunk_diversity_loss


def compute_total_loss(
    logits: mx.array,
    targets: mx.array,
    retrieval_scores: mx.array,
    chunk_indices: mx.array,
    target_chunk_indices: mx.array,
    index_keys: mx.array,
    centroids: mx.array,
    pad_id: int = 0,
    lambda_retrieval: float = 0.5,
    lambda_reg: float = 0.01,
    all_chunk_scores: mx.array = None,
) -> Tuple[mx.array, Dict[str, float]]:
    """Compute total loss with all components.

    Args:
        logits: Model output logits.
        targets: Target token IDs.
        retrieval_scores: Retrieval relevance scores.
        chunk_indices: Retrieved chunk indices.
        target_chunk_indices: Ground truth chunk indices.
        index_keys: Index keys for regularization.
        centroids: Cluster centroids for regularization.
        pad_id: Padding token ID.
        lambda_retrieval: Weight for retrieval loss.
        lambda_reg: Weight for regularization loss.
        all_chunk_scores: Optional scores for ALL chunks for direct supervision.

    Returns:
        total_loss: Scalar loss for optimization.
        metrics: Dictionary of individual loss components.
    """
    gen_loss = compute_generation_loss(logits, targets, pad_id)
    ret_loss = compute_retrieval_loss(
        retrieval_scores, chunk_indices, target_chunk_indices, all_chunk_scores
    )
    reg_loss = compute_index_regularization(index_keys, centroids)

    total = gen_loss + lambda_retrieval * ret_loss + lambda_reg * reg_loss

    # Evaluate to get scalar values for metrics
    metrics = {
        "generation_loss": float(gen_loss),
        "retrieval_loss": float(ret_loss),
        "regularization_loss": float(reg_loss),
        "total_loss": float(total),
    }

    return total, metrics


def compute_accuracy(
    logits: mx.array,
    targets: mx.array,
    pad_id: int = 0,
) -> float:
    """Compute token-level accuracy.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        pad_id: Padding token ID to ignore.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    predictions = mx.argmax(logits, axis=-1)
    correct = predictions == targets
    mask = targets != pad_id
    accuracy = (correct * mask).sum() / (mask.sum() + 1e-8)
    return float(accuracy)


def compute_retrieval_recall(
    chunk_indices: mx.array,
    target_chunk_indices: mx.array,
) -> float:
    """Compute retrieval recall (fraction of targets in top-k).

    Args:
        chunk_indices: Retrieved chunk indices of shape (batch, top_k).
        target_chunk_indices: Ground truth chunk indices of shape (batch,).

    Returns:
        Recall as a float between 0 and 1.
    """
    target_expanded = target_chunk_indices[:, None]
    is_retrieved = mx.any(chunk_indices == target_expanded, axis=-1)
    return float(mx.mean(is_retrieved.astype(mx.float32)))
