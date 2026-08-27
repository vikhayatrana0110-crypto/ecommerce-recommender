"""Top-N ranking metrics.

Every function takes an ordered `recommended` sequence of item indices and a
`relevant` set of held-out item indices, and returns a score in [0, 1].
"""
import numpy as np


def precision_at_k(recommended, relevant, k):
    """Fraction of the top-k slots filled by a relevant item."""
    if k <= 0:
        return 0.0
    top_k = list(recommended)[:k]
    return len(set(top_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    """Fraction of the relevant items retrieved within the top-k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    return len(set(top_k) & relevant) / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """Precision averaged over the ranks at which relevant items appear.

    Normalised by min(|relevant|, k), the best achievable number of hits.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0

    hits, score = 0, 0.0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank

    denom = min(len(relevant), k)
    return score / denom if denom else 0.0


def map_at_k(recommendations, relevants, k):
    """Mean average precision across users."""
    scores = [
        average_precision_at_k(rec, rel, k)
        for rec, rel in zip(recommendations, relevants)
    ]
    return float(np.mean(scores)) if scores else 0.0


def dcg_at_k(recommended, relevant, k):
    """Discounted cumulative gain with binary relevance."""
    relevant = set(relevant)
    gains = [
        1.0 / np.log2(rank + 1)
        for rank, item in enumerate(list(recommended)[:k], start=1)
        if item in relevant
    ]
    return float(sum(gains))


def ndcg_at_k(recommended, relevant, k):
    """DCG normalised by the best ordering achievable for this user."""
    relevant = set(relevant)
    if not relevant:
        return 0.0

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg


def evaluate_rankings(recommendations, relevants, k):
    """Aggregates all four metrics over aligned recommendation/relevance lists."""
    if not recommendations:
        return {f"Precision@{k}": 0.0, f"Recall@{k}": 0.0,
                f"NDCG@{k}": 0.0, f"MAP@{k}": 0.0}

    pairs = list(zip(recommendations, relevants))
    return {
        f"Precision@{k}": float(np.mean([precision_at_k(r, t, k) for r, t in pairs])),
        f"Recall@{k}": float(np.mean([recall_at_k(r, t, k) for r, t in pairs])),
        f"NDCG@{k}": float(np.mean([ndcg_at_k(r, t, k) for r, t in pairs])),
        f"MAP@{k}": map_at_k(recommendations, relevants, k),
    }
