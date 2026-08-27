import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.eval.split import build_split, random_split, temporal_leave_last_out


@pytest.fixture
def matrix():
    return csr_matrix(np.array([
        [5, 4, 0, 1],
        [0, 3, 4, 5],
        [1, 0, 5, 4],
    ], dtype=float))


def test_random_split_is_deterministic_under_a_seed(matrix):
    a1, b1 = random_split(matrix, 0.5, seed=42)
    a2, b2 = random_split(matrix, 0.5, seed=42)
    assert (a1 != a2).nnz == 0
    assert (b1 != b2).nnz == 0


def test_random_split_differs_across_seeds(matrix):
    _, b1 = random_split(matrix, 0.5, seed=1)
    _, b2 = random_split(matrix, 0.5, seed=999)
    assert (b1 != b2).nnz > 0


def test_random_split_preserves_every_interaction(matrix):
    train, test = random_split(matrix, 0.5, seed=7)
    assert np.allclose((train + test).toarray(), matrix.toarray())


def test_random_split_has_no_leakage(matrix):
    train, test = random_split(matrix, 0.5, seed=7)
    assert train.multiply(test).nnz == 0


def test_random_split_keeps_single_interaction_users_in_train():
    single = csr_matrix(np.array([[1, 0, 0]], dtype=float))
    train, test = random_split(single, 0.5, seed=3)
    assert train.nnz == 1 and test.nnz == 0


@pytest.fixture
def temporal_df():
    return pd.DataFrame({
        "user_id": ["U1", "U1", "U1", "U2", "U2", "U3"],
        "item_id": ["A", "B", "C", "A", "D", "B"],
        "rating": [1.0] * 6,
        "timestamp": [1, 2, 3, 1, 2, 1],
    })


MAPS = ({"U1": 0, "U2": 1, "U3": 2}, {"A": 0, "B": 1, "C": 2, "D": 3})


def test_temporal_split_holds_out_the_most_recent_item(temporal_df):
    train, test = temporal_leave_last_out(temporal_df, *MAPS)
    assert test[0, 2] == 1.0   # U1's latest is C
    assert test[1, 3] == 1.0   # U2's latest is D
    assert test.nnz == 2


def test_temporal_split_keeps_single_interaction_users_in_train(temporal_df):
    train, test = temporal_leave_last_out(temporal_df, *MAPS)
    assert train[2].nnz == 1 and test[2].nnz == 0


def test_temporal_split_has_no_leakage(temporal_df):
    train, test = temporal_leave_last_out(temporal_df, *MAPS)
    assert train.multiply(test).nnz == 0


def test_temporal_split_does_not_train_on_the_future(temporal_df):
    """U1's held-out item C must not appear anywhere in the training row."""
    train, _ = temporal_leave_last_out(temporal_df, *MAPS)
    assert 2 not in set(train[0].indices)


def test_temporal_split_requires_timestamps():
    df = pd.DataFrame({"user_id": ["U1"], "item_id": ["A"], "rating": [1.0]})
    with pytest.raises(KeyError):
        temporal_leave_last_out(df, {"U1": 0}, {"A": 0})


def test_build_split_dispatches_on_strategy(temporal_df, matrix):
    cfg = {"split": {"strategy": "temporal"}, "seed": 1}
    _, test = build_split(temporal_df, matrix, *MAPS, cfg)
    assert test.nnz == 2

    cfg = {"split": {"strategy": "random", "test_ratio": 0.5}, "seed": 1}
    train, test = build_split(temporal_df, matrix, *MAPS, cfg)
    assert np.allclose((train + test).toarray(), matrix.toarray())
