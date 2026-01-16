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
) -> mx.array:
    """Regularization to encourage diverse, well-separated keys.

    Two objectives:
    1. Keys within each chunk should be diverse (capture different info)
    2. Chunk-level key representations should be diverse (different chunks different)

    Args:
        index_keys: Keys of shape (num_chunks, keys_per_chunk, index_dim).

    Returns:
        Scalar regularization loss.
    """
    num_chunks = index_keys.shape[0]
    keys_per_chunk = index_keys.shape[1]

    # Loss 1: Keys within each chunk should be diverse
    # Normalize keys for cosine similarity
    key_norms = mx.linalg.norm(index_keys, axis=-1, keepdims=True)
    normalized_keys = index_keys / (key_norms + 1e-8)

    # Compute within-chunk key similarity
    # For each chunk, compute pairwise similarity between its keys
    intra_chunk_loss = mx.array(0.0)
    if keys_per_chunk > 1:
        for i in range(num_chunks):
            chunk_keys = normalized_keys[i]  # (keys_per_chunk, index_dim)
            key_sim = mx.matmul(chunk_keys, chunk_keys.T)  # (keys_per_chunk, keys_per_chunk)
            # Mask diagonal
            mask = 1 - mx.eye(keys_per_chunk)
            intra_chunk_loss = intra_chunk_loss + mx.mean(key_sim * mask)
        intra_chunk_loss = intra_chunk_loss / num_chunks

    # Loss 2: Chunk-level keys should be diverse (different chunks different)
    # Average keys per chunk to get chunk-level representation
    chunk_key_repr = mx.mean(index_keys, axis=1)  # (num_chunks, index_dim)
    chunk_key_norm = chunk_key_repr / (mx.linalg.norm(chunk_key_repr, axis=-1, keepdims=True) + 1e-8)
    chunk_sim = mx.matmul(chunk_key_norm, chunk_key_norm.T)  # (num_chunks, num_chunks)

    # Penalize high similarity between different chunks
    chunk_mask = 1 - mx.eye(num_chunks)
    chunk_diversity_loss = mx.mean(chunk_sim * chunk_mask)

    return 0.1 * intra_chunk_loss + 0.5 * chunk_diversity_loss


def compute_total_loss(
    logits: mx.array,
    targets: mx.array,
    retrieval_scores: mx.array,
    chunk_indices: mx.array,
    target_chunk_indices: mx.array,
    index_keys: mx.array,
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
    reg_loss = compute_index_regularization(index_keys)

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


def compute_contrastive_embedding_loss(
    query_tokens: mx.array,
    chunk_tokens: mx.array,
    target_chunk_indices: mx.array,
    shared_embedding: mx.array,
    temperature: float = 0.1,
) -> mx.array:
    """Contrastive loss to align query embeddings with correct chunk embeddings.

    Uses token overlap as a soft supervision signal to bootstrap retrieval learning.
    The loss encourages the mean query embedding to be similar to the mean embedding
    of the correct chunk.

    Args:
        query_tokens: Query token IDs of shape (batch, query_len).
        chunk_tokens: All chunk token IDs of shape (num_chunks, chunk_size).
        target_chunk_indices: Ground truth chunk indices of shape (batch,).
        shared_embedding: Shared embedding weight matrix of shape (vocab_size, d_model).
        temperature: Temperature for contrastive loss.

    Returns:
        Scalar contrastive loss.
    """
    batch_size = query_tokens.shape[0]
    num_chunks = chunk_tokens.shape[0]

    # Get query embeddings and compute mean
    query_embed = shared_embedding[query_tokens]  # (batch, query_len, d_model)
    query_mean = mx.mean(query_embed, axis=1)  # (batch, d_model)

    # Get chunk embeddings and compute mean
    chunk_embed = shared_embedding[chunk_tokens]  # (num_chunks, chunk_size, d_model)
    chunk_mean = mx.mean(chunk_embed, axis=1)  # (num_chunks, d_model)

    # Normalize for cosine similarity
    query_norm = query_mean / (mx.linalg.norm(query_mean, axis=-1, keepdims=True) + 1e-8)
    chunk_norm = chunk_mean / (mx.linalg.norm(chunk_mean, axis=-1, keepdims=True) + 1e-8)

    # Compute similarity between all queries and all chunks
    similarity = mx.matmul(query_norm, chunk_norm.T) / temperature  # (batch, num_chunks)

    # Cross-entropy loss: target chunk should have highest similarity
    log_probs = mx.log(mx.softmax(similarity, axis=-1) + 1e-10)
    batch_indices = mx.arange(batch_size)
    target_log_probs = log_probs[batch_indices, target_chunk_indices]

    return -mx.mean(target_log_probs)


def compute_copy_gate_loss(
    copy_gate: mx.array,
    target_copy_prob: float = 0.8,
) -> mx.array:
    """Loss to encourage appropriate copy gate behavior.

    For HashHop, we expect the model to mostly copy from context.
    This loss provides a soft prior that copy_gate should be high.

    Args:
        copy_gate: Copy gate values of shape (batch, seq_len, 1).
        target_copy_prob: Target probability for copying (default 0.8).

    Returns:
        Scalar loss value.
    """
    # Binary cross-entropy with target_copy_prob
    # Loss = -[target * log(gate) + (1-target) * log(1-gate)]
    target = mx.array(target_copy_prob)
    gate = copy_gate.squeeze(-1)  # (batch, seq_len)

    loss = -target * mx.log(gate + 1e-8) - (1 - target) * mx.log(1 - gate + 1e-8)
    return mx.mean(loss)


def compute_copy_attention_loss(
    copy_attention: mx.array,
    copy_targets: mx.array,
    target_chunk_indices: mx.array,
    top_k: int,
    chunk_indices: mx.array,
) -> mx.array:
    """Loss to supervise where copy attention should focus at each position.

    This provides position-specific supervision: for each output position i,
    tells the copy attention which context position contains the token to copy.

    Only applies loss when the target chunk is actually in the retrieved chunks.
    When retrieval fails, we can't provide meaningful copy supervision.

    Args:
        copy_attention: Copy attention weights (batch, seq_len, context_len).
        copy_targets: Position-specific target mask (batch, seq_len, chunk_size).
            copy_targets[b, i, j] = 1.0 if output position i should copy
            from chunk position j.
        target_chunk_indices: Which chunk contains the answer (batch,).
        top_k: Number of retrieved chunks.
        chunk_indices: Retrieved chunk indices (batch, top_k).

    Returns:
        Scalar loss value.
    """
    batch_size, seq_len, context_len = copy_attention.shape
    chunk_size = context_len // top_k

    # Find which position in retrieved chunks contains the target chunk
    # chunk_indices is (batch, top_k), target_chunk_indices is (batch,)
    target_expanded = target_chunk_indices[:, None]  # (batch, 1)
    is_target_chunk = (chunk_indices == target_expanded).astype(mx.float32)  # (batch, top_k)

    # Check if target chunk is in retrieved chunks for each batch element
    target_retrieved = mx.any(chunk_indices == target_expanded, axis=-1).astype(mx.float32)  # (batch,)

    # Find which retrieved chunk index (0 to top_k-1) is the target
    # This tells us where to place the copy targets in the full context
    target_chunk_position = mx.argmax(is_target_chunk, axis=-1)  # (batch,)

    # Build full context target mask
    # copy_targets is (batch, seq_len, chunk_size) - positions within the target chunk
    # We need to expand to (batch, seq_len, context_len) placing at correct chunk position

    # Initialize full target mask
    full_target_mask = mx.zeros((batch_size, seq_len, context_len))

    # For each possible target chunk position, add the copy targets at that offset
    for k in range(top_k):
        # Mask for samples where target is at position k
        is_at_k = (target_chunk_position == k).astype(mx.float32)  # (batch,)

        # Offset in the full context for chunk k
        offset = k * chunk_size

        # Create the full context mask for this chunk position
        # copy_targets: (batch, seq_len, chunk_size)
        # Need to place at offset in the context dimension

        # Weight by is_at_k: (batch, 1, 1) * (batch, seq_len, chunk_size)
        weighted = is_at_k[:, None, None] * copy_targets

        # Build indices for scatter - this is the tricky part
        # We need to add to positions [offset:offset+chunk_size] in context dimension

        # Concatenate zeros before and after
        if offset > 0:
            zeros_before = mx.zeros((batch_size, seq_len, offset))
        else:
            zeros_before = None

        remaining = context_len - offset - chunk_size
        if remaining > 0:
            zeros_after = mx.zeros((batch_size, seq_len, remaining))
        else:
            zeros_after = None

        # Build slice for this chunk
        parts = []
        if zeros_before is not None:
            parts.append(zeros_before)
        parts.append(weighted)
        if zeros_after is not None:
            parts.append(zeros_after)

        chunk_mask = mx.concatenate(parts, axis=-1)  # (batch, seq_len, context_len)
        full_target_mask = full_target_mask + chunk_mask

    # Check which positions have valid targets (non-zero sum)
    position_has_target = (mx.sum(full_target_mask, axis=-1) > 0).astype(mx.float32)  # (batch, seq_len)

    # Normalize target mask to be a probability distribution per position
    target_sum = mx.sum(full_target_mask, axis=-1, keepdims=True)  # (batch, seq_len, 1)
    full_target_mask = full_target_mask / (target_sum + 1e-8)

    # Cross-entropy loss between copy attention and target distribution
    log_attn = mx.log(copy_attention + 1e-8)  # (batch, seq_len, context_len)
    ce_loss = -mx.sum(full_target_mask * log_attn, axis=-1)  # (batch, seq_len)

    # Only apply loss when:
    # 1. Target was retrieved
    # 2. This position has a valid target
    valid_mask = target_retrieved[:, None] * position_has_target  # (batch, seq_len)
    ce_loss = ce_loss * valid_mask

    # Average over valid positions only
    num_valid = mx.sum(valid_mask) + 1e-8
    return mx.sum(ce_loss) / num_valid


def compute_total_loss_with_copy(
    logits: mx.array,
    targets: mx.array,
    copy_gate: mx.array,
    copy_attention: mx.array,
    copy_targets: mx.array,
    retrieval_scores: mx.array,
    chunk_indices: mx.array,
    target_chunk_indices: mx.array,
    index_keys: mx.array,
    top_k: int,
    pad_id: int = 0,
    lambda_retrieval: float = 2.0,
    lambda_reg: float = 0.01,
    lambda_copy: float = 0.1,
    lambda_copy_attn: float = 1.0,
    all_chunk_scores: mx.array = None,
) -> Tuple[mx.array, Dict[str, float]]:
    """Compute total loss for autoregressive model with copy mechanism.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len).
        copy_gate: Copy gate values of shape (batch, seq_len, 1).
        copy_attention: Copy attention weights (batch, seq_len, context_len).
        copy_targets: Target mask for copy positions (batch, chunk_size).
        retrieval_scores: Retrieval relevance scores.
        chunk_indices: Retrieved chunk indices.
        target_chunk_indices: Ground truth chunk indices.
        index_keys: Index keys for regularization.
        top_k: Number of retrieved chunks.
        pad_id: Padding token ID.
        lambda_retrieval: Weight for retrieval loss.
        lambda_reg: Weight for regularization loss.
        lambda_copy: Weight for copy gate loss.
        lambda_copy_attn: Weight for copy attention supervision loss.
        all_chunk_scores: Optional scores for ALL chunks for direct supervision.

    Returns:
        total_loss: Scalar loss for optimization.
        metrics: Dictionary of individual loss components.
    """
    gen_loss = compute_generation_loss(logits, targets, pad_id)
    ret_loss = compute_retrieval_loss(
        retrieval_scores, chunk_indices, target_chunk_indices, all_chunk_scores
    )
    reg_loss = compute_index_regularization(index_keys)
    copy_gate_loss = compute_copy_gate_loss(copy_gate)

    # Copy attention supervision - teach the model WHERE to attend
    copy_attn_loss = compute_copy_attention_loss(
        copy_attention, copy_targets, target_chunk_indices, top_k, chunk_indices
    )

    total = (
        gen_loss
        + lambda_retrieval * ret_loss
        + lambda_reg * reg_loss
        + lambda_copy * copy_gate_loss
        + lambda_copy_attn * copy_attn_loss
    )

    # Evaluate to get scalar values for metrics
    metrics = {
        "generation_loss": float(gen_loss),
        "retrieval_loss": float(ret_loss),
        "regularization_loss": float(reg_loss),
        "copy_gate_loss": float(copy_gate_loss),
        "copy_attn_loss": float(copy_attn_loss),
        "copy_gate_mean": float(mx.mean(copy_gate)),
        "total_loss": float(total),
    }

    return total, metrics
