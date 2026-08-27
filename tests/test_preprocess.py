import pandas as pd
import pytest

from src.data.preprocess import (
    binarize_ratings,
    create_interaction_matrix,
    dedupe_interactions,
    describe,
    iterative_filter,
)


def test_dedupe_keeps_latest_review_per_user_item():
    df = pd.DataFrame({
        "user_id": ["U1", "U1", "U2"],
        "item_id": ["I1", "I1", "I1"],
        "rating": [3.0, 5.0, 4.0],
        "timestamp": [100, 200, 100],
    })
    out = dedupe_interactions(df)
    assert len(out) == 2
    # The later (timestamp 200) review wins.
    assert out[out["user_id"] == "U1"]["rating"].iloc[0] == 5.0


def test_dedupe_without_timestamp_column():
    df = pd.DataFrame({
        "user_id": ["U1", "U1"], "item_id": ["I1", "I1"], "rating": [3.0, 5.0],
    })
    assert len(dedupe_interactions(df)) == 1


def test_duplicates_would_otherwise_inflate_the_matrix():
    """Without dedupe, csr_matrix sums repeats into an impossible rating."""
    df = pd.DataFrame({
        "user_id": ["U1", "U1"], "item_id": ["I1", "I1"],
        "rating": [5.0, 5.0], "timestamp": [1, 2],
    })
    inflated, _, _ = create_interaction_matrix(df)
    assert inflated[0, 0] == 10.0

    clean, _, _ = create_interaction_matrix(dedupe_interactions(df))
    assert clean[0, 0] == 5.0


def _chain_df():
    """U3 survives only because of I2, which is itself only kept by U3."""
    rows = []
    for user in ("U1", "U2"):
        for item in ("I1", "I2", "I3"):
            rows.append((user, item))
    rows.append(("U3", "I2"))
    return pd.DataFrame({
        "user_id": [r[0] for r in rows],
        "item_id": [r[1] for r in rows],
        "rating": [1.0] * len(rows),
    })


def test_iterative_filter_reaches_a_stable_k_core():
    out = iterative_filter(_chain_df(), min_reviews_user=2, min_reviews_item=2)
    # U3 has a single interaction and must be gone.
    assert "U3" not in set(out["user_id"])
    # What remains genuinely satisfies both thresholds.
    assert out.groupby("user_id").size().min() >= 2
    assert out.groupby("item_id").size().min() >= 2


def test_iterative_filter_is_idempotent():
    once = iterative_filter(_chain_df(), 2, 2)
    twice = iterative_filter(once, 2, 2)
    assert len(once) == len(twice)


def test_iterative_filter_can_empty_a_sparse_frame():
    df = pd.DataFrame({
        "user_id": ["U1", "U2"], "item_id": ["I1", "I2"], "rating": [1.0, 1.0],
    })
    assert iterative_filter(df, min_reviews_user=5, min_reviews_item=5).empty


def test_binarize_drops_low_ratings_and_flattens_the_rest():
    df = pd.DataFrame({
        "user_id": ["U1", "U2", "U3"], "item_id": ["I1"] * 3, "rating": [5.0, 4.0, 2.0],
    })
    out = binarize_ratings(df, threshold=4.0)
    assert len(out) == 2
    assert set(out["rating"]) == {1.0}


def test_create_interaction_matrix_honours_supplied_maps():
    """Serving must reuse the trained mapping or the factors mean nothing."""
    df = pd.DataFrame({
        "user_id": ["U1", "U2"], "item_id": ["I1", "I2"], "rating": [5.0, 4.0],
    })
    user_map = {"U2": 0, "U1": 1}
    item_map = {"I2": 0, "I1": 1}
    matrix, um, im = create_interaction_matrix(df, user_map=user_map, item_map=item_map)

    assert um is user_map and im is item_map
    assert matrix[0, 0] == 4.0   # U2/I2 at the supplied indices
    assert matrix[1, 1] == 5.0   # U1/I1


def test_create_interaction_matrix_drops_unknown_ids():
    df = pd.DataFrame({
        "user_id": ["U1", "GHOST"], "item_id": ["I1", "I1"], "rating": [5.0, 1.0],
    })
    matrix, _, _ = create_interaction_matrix(df, user_map={"U1": 0}, item_map={"I1": 0})
    assert matrix.shape == (1, 1)
    assert matrix.nnz == 1


def test_describe_reports_density():
    df = pd.DataFrame({
        "user_id": ["U1", "U1", "U2"], "item_id": ["I1", "I2", "I1"], "rating": [1.0] * 3,
    })
    stats = describe(df, "test")
    assert stats["users"] == 2 and stats["items"] == 2
    assert stats["interactions"] == 3
    assert stats["density"] == pytest.approx(0.75)
    assert stats["per_user"] == pytest.approx(1.5)
