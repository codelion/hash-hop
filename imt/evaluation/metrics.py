"""Evaluation metrics for IMT on HashHop tasks."""

from dataclasses import dataclass
from typing import Dict, List

import mlx.core as mx


@dataclass
class HopMetrics:
    """Metrics broken down by hop count."""

    accuracy_by_hop: Dict[int, float]
    retrieval_recall_by_hop: Dict[int, float]
    average_retrieval_rank_by_hop: Dict[int, float]


def compute_hop_metrics(
    predictions: List[str],
    targets: List[str],
    retrieved_chunks: List[List[int]],
    target_chunks: List[int],
    hop_counts: List[int],
) -> HopMetrics:
    """Compute accuracy metrics broken down by number of hops.

    For HashHop, we can test:
    - 1-hop: Direct key-value lookup
    - 2-hop: Key -> intermediate -> value
    - 3-hop: Key -> int1 -> int2 -> value

    Args:
        predictions: List of predicted hash strings.
        targets: List of target hash strings.
        retrieved_chunks: List of retrieved chunk index lists for each query.
        target_chunks: List of ground truth chunk indices.
        hop_counts: List of hop counts for each query.

    Returns:
        HopMetrics with per-hop accuracy breakdowns.
    """
    accuracy_by_hop: Dict[int, List[int]] = {}
    retrieval_recall_by_hop: Dict[int, List[int]] = {}
    retrieval_rank_by_hop: Dict[int, List[int]] = {}

    for i, hops in enumerate(hop_counts):
        if hops not in accuracy_by_hop:
            accuracy_by_hop[hops] = []
            retrieval_recall_by_hop[hops] = []
            retrieval_rank_by_hop[hops] = []

        # Accuracy
        correct = 1 if predictions[i] == targets[i] else 0
        accuracy_by_hop[hops].append(correct)

        # Retrieval recall
        recalled = 1 if target_chunks[i] in retrieved_chunks[i] else 0
        retrieval_recall_by_hop[hops].append(recalled)

        # Retrieval rank
        if target_chunks[i] in retrieved_chunks[i]:
            rank = retrieved_chunks[i].index(target_chunks[i])
            retrieval_rank_by_hop[hops].append(rank)

    # Compute averages
    return HopMetrics(
        accuracy_by_hop={
            h: sum(v) / len(v) if v else 0 for h, v in accuracy_by_hop.items()
        },
        retrieval_recall_by_hop={
            h: sum(v) / len(v) if v else 0 for h, v in retrieval_recall_by_hop.items()
        },
        average_retrieval_rank_by_hop={
            h: sum(v) / len(v) if v else -1 for h, v in retrieval_rank_by_hop.items()
        },
    )


def compute_exact_match_accuracy(
    predictions: List[str],
    targets: List[str],
) -> float:
    """Compute exact match accuracy.

    Args:
        predictions: List of predicted strings.
        targets: List of target strings.

    Returns:
        Accuracy as float between 0 and 1.
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def compute_character_accuracy(
    predictions: List[str],
    targets: List[str],
) -> float:
    """Compute character-level accuracy.

    Args:
        predictions: List of predicted strings.
        targets: List of target strings.

    Returns:
        Character-level accuracy as float between 0 and 1.
    """
    total_chars = 0
    correct_chars = 0

    for pred, target in zip(predictions, targets):
        # Pad shorter string
        max_len = max(len(pred), len(target))
        pred_padded = pred.ljust(max_len)
        target_padded = target.ljust(max_len)

        for p, t in zip(pred_padded, target_padded):
            total_chars += 1
            if p == t:
                correct_chars += 1

    return correct_chars / total_chars if total_chars > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_chunks: List[List[int]],
    target_chunks: List[int],
) -> Dict[str, float]:
    """Compute retrieval-specific metrics.

    Args:
        retrieved_chunks: List of retrieved chunk index lists.
        target_chunks: List of target chunk indices.

    Returns:
        Dictionary with recall@k metrics.
    """
    n = len(target_chunks)
    if n == 0:
        return {"recall@1": 0.0, "recall@4": 0.0, "mrr": 0.0}

    recall_1 = 0
    recall_4 = 0
    reciprocal_ranks = []

    for retrieved, target in zip(retrieved_chunks, target_chunks):
        # Recall@1
        if len(retrieved) > 0 and retrieved[0] == target:
            recall_1 += 1

        # Recall@4
        if target in retrieved[:4]:
            recall_4 += 1

        # MRR (Mean Reciprocal Rank)
        if target in retrieved:
            rank = retrieved.index(target) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "recall@1": recall_1 / n,
        "recall@4": recall_4 / n,
        "mrr": sum(reciprocal_ranks) / n,
    }
