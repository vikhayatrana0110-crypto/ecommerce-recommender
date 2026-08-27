import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.models.als import ALSRecommender
from src.models.content import ContentRecommender, cold_start_recommend
from src.models.popularity import PopularityRecommender


@pytest.fixture
def matrix():
    # Item 2 is the most popular, item 3 the least.
    return csr_matrix(np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=float))


# ------------------------------------------------------------------ popularity
def test_popularity_ranks_by_total_interactions(matrix):
    model = PopularityRecommender().fit(matrix)
    assert model.ranking[0] == 2
    assert list(model.recommend(2)) == [2, 0] or list(model.recommend(2))[0] == 2


def test_popularity_excludes_seen_items(matrix):
    model = PopularityRecommender().fit(matrix)
    assert 2 not in set(model.recommend(2, exclude=[2]))


def test_popularity_bounded_window_matches_full_scan(matrix):
    """The fast path must agree with an exhaustive scan."""
    model = PopularityRecommender().fit(matrix)
    exclude = [2, 0]
    fast = list(model.recommend(2, exclude=exclude))
    slow = [i for i in model.ranking if i not in set(exclude)][:2]
    assert fast == slow


def test_popularity_batch_filters_each_users_history(matrix):
    model = PopularityRecommender().fit(matrix)
    recs = model.recommend_batch([0, 1, 2], matrix, n=2)
    for user, rec in enumerate(recs):
        assert not (set(int(i) for i in rec) & set(matrix[user].indices))


def test_popularity_requires_fit():
    with pytest.raises(RuntimeError):
        PopularityRecommender().recommend(5)


# --------------------------------------------------------------------- content
@pytest.fixture
def metadata():
    return pd.DataFrame({
        "item_id": ["A", "B", "C", "D"],
        "title": [
            "Wireless Bluetooth Headphones Over Ear",
            "Bluetooth Headphones Wireless Earbuds",
            "USB C Charging Cable Braided",
            "Stainless Steel Kitchen Blender",
        ],
        "categories": [["Audio", "Headphones"], ["Audio", "Headphones"],
                       ["Cables"], ["Appliances"]],
        "main_category": ["Electronics"] * 4,
    })


ITEM_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def test_content_finds_the_semantically_closest_item(metadata):
    model = ContentRecommender(min_df=1).fit(metadata, ITEM_MAP)
    # The two headphone listings should be each other's nearest neighbour.
    assert model.similar_items(0, n=1)[0] == 1
    assert model.similar_items(1, n=1)[0] == 0


def test_content_never_recommends_the_seed_item(metadata):
    model = ContentRecommender(min_df=1).fit(metadata, ITEM_MAP)
    assert 0 not in set(model.similar_items(0, n=3))


def test_content_profile_excludes_already_seen(metadata):
    model = ContentRecommender(min_df=1).fit(metadata, ITEM_MAP)
    assert 0 not in set(model.recommend_from_profile([0], n=3))


def test_content_aligns_rows_with_the_model_index_space(metadata):
    """Row i must be item index i, not metadata row order."""
    shuffled = metadata.iloc[::-1].reset_index(drop=True)
    model = ContentRecommender(min_df=1).fit(shuffled, ITEM_MAP)
    assert model.similar_items(0, n=1)[0] == 1


def test_content_disabled_when_metadata_is_empty():
    empty = pd.DataFrame(columns=["item_id", "title", "categories", "main_category"])
    model = ContentRecommender(min_df=1).fit(empty, ITEM_MAP)
    assert not model.available
    assert len(model.similar_items(0, n=3)) == 0


# ------------------------------------------------------------------ cold start
def test_cold_start_uses_content_when_history_exists(metadata, matrix):
    content = ContentRecommender(min_df=1).fit(metadata, ITEM_MAP)
    popularity = PopularityRecommender().fit(matrix)
    recs = cold_start_recommend(content, popularity, [0], n=3)
    assert recs[0] == 1          # nearest neighbour of the headphone item
    assert 0 not in set(recs)


def test_cold_start_falls_back_to_popularity_without_history(metadata, matrix):
    content = ContentRecommender(min_df=1).fit(metadata, ITEM_MAP)
    popularity = PopularityRecommender().fit(matrix)
    recs = cold_start_recommend(content, popularity, [], n=2)
    assert recs[0] == popularity.ranking[0]


def test_cold_start_always_returns_n_items(metadata, matrix):
    content = ContentRecommender(min_df=1).fit(metadata, ITEM_MAP)
    popularity = PopularityRecommender().fit(matrix)
    assert len(cold_start_recommend(content, popularity, [0], n=4)) == 4


def test_cold_start_works_without_a_content_model(matrix):
    popularity = PopularityRecommender().fit(matrix)
    recs = cold_start_recommend(None, popularity, [0], n=2)
    assert len(recs) == 2


# ------------------------------------------------------------------------- ALS
def test_als_trains_and_recommends_and_round_trips(tmp_path, matrix):
    model = ALSRecommender(factors=4, iterations=2, seed=0).fit(matrix)

    recs = model.recommend_batch([0, 1], matrix, n=2)
    assert len(recs) == 2 and len(recs[0]) == 2

    path = tmp_path / "model.pkl"
    model.save(path)
    reloaded = ALSRecommender.load(path)
    assert np.array_equal(
        reloaded.recommend_batch([0], matrix, n=2)[0],
        recs[0],
    )


def test_als_filters_already_seen_items(matrix):
    model = ALSRecommender(factors=4, iterations=2, seed=0).fit(matrix)
    recs = model.recommend_batch([0], matrix, n=2)[0]
    assert not (set(int(i) for i in recs) & set(matrix[0].indices))
