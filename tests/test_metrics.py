import numpy as np
import pytest

from src.eval.metrics import (
    average_precision_at_k,
    dcg_at_k,
    evaluate_rankings,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_and_recall_basic():
    # 2 of the top 5 recommendations are relevant, and they are the only 2.
    assert precision_at_k([1, 2, 3, 4, 5], {1, 2}, 5) == pytest.approx(0.4)
    assert recall_at_k([1, 2, 3, 4, 5], {1, 2}, 5) == pytest.approx(1.0)


def test_precision_uses_k_not_list_length():
    # Only 3 recommendations returned but k=10: precision is still out of 10.
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 10) == pytest.approx(0.3)


def test_metrics_are_zero_without_relevant_items():
    assert recall_at_k([1, 2], set(), 5) == 0.0
    assert ndcg_at_k([1, 2], set(), 5) == 0.0
    assert average_precision_at_k([1, 2], set(), 5) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k([1, 2, 9, 8], {1, 2}, 4) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation():
    # Relevant items land at ranks 2 and 4.
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert ndcg_at_k([9, 1, 8, 2, 7], {1, 2}, 5) == pytest.approx(dcg / idcg)


def test_ndcg_rewards_higher_ranks():
    early = ndcg_at_k([1, 9, 8, 7], {1}, 4)
    late = ndcg_at_k([9, 8, 7, 1], {1}, 4)
    assert early > late


def test_dcg_ignores_items_beyond_k():
    assert dcg_at_k([9, 8, 1], {1}, 2) == 0.0


def test_average_precision_matches_hand_computation():
    # Hits at ranks 2 and 4 -> (1/2 + 2/4) / 2
    assert average_precision_at_k([9, 1, 8, 2], {1, 2}, 4) == pytest.approx(0.5)


def test_average_precision_normalises_by_reachable_hits():
    # 5 relevant items but only k=2 slots: perfect ranking must score 1.0.
    assert average_precision_at_k([1, 2], {1, 2, 3, 4, 5}, 2) == pytest.approx(1.0)


def test_map_averages_across_users():
    perfect = ([1, 2], {1, 2})
    miss = ([8, 9], {1, 2})
    assert map_at_k([perfect[0], miss[0]], [perfect[1], miss[1]], 2) == pytest.approx(0.5)


def test_evaluate_rankings_returns_all_four_metrics():
    results = evaluate_rankings([[1, 2, 3]], [{1}], 3)
    assert set(results) == {"Precision@3", "Recall@3", "NDCG@3", "MAP@3"}
    assert all(0.0 <= v <= 1.0 for v in results.values())


def test_evaluate_rankings_handles_empty_input():
    results = evaluate_rankings([], [], 10)
    assert all(v == 0.0 for v in results.values())
