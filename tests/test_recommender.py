import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from src.data.preprocess import clean_reviews, filter_active_users, filter_popular_items, create_interaction_matrix
from main import train_test_split

@pytest.fixture
def sample_reviews_df():
    data = {
        "user_id": ["U1", "U1", "U2", "U2", "U3", None],
        "item_id": ["I1", "I2", "I1", "I2", "I1", "I2"],
        "rating": [5, 4, 3, 5, 0, 4]
    }
    return pd.DataFrame(data)

def test_clean_reviews(sample_reviews_df):
    cleaned = clean_reviews(sample_reviews_df)
    # Check null user_id is dropped
    assert cleaned["user_id"].isnull().sum() == 0
    # Check rating <= 0 is filtered out (rating=0 should be removed)
    assert (cleaned["rating"] <= 0).sum() == 0
    assert len(cleaned) == 4

def test_filter_active_users(sample_reviews_df):
    cleaned = clean_reviews(sample_reviews_df)
    # U1 and U2 have 2 reviews, U3 has 0 after cleaning. Filter users with min_reviews >= 2.
    filtered = filter_active_users(cleaned, min_reviews=2)
    assert set(filtered["user_id"].unique()) == {"U1", "U2"}

def test_create_interaction_matrix(sample_reviews_df):
    cleaned = clean_reviews(sample_reviews_df)
    matrix, user_map, item_map = create_interaction_matrix(cleaned)
    assert matrix.shape == (2, 2)
    assert user_map == {"U1": 0, "U2": 1}
    assert item_map == {"I1": 0, "I2": 1}
    assert matrix[0, 0] == 5.0

def test_train_test_split():
    # Build a simple dense matrix for split testing
    np.random.seed(42)
    interaction_matrix = csr_matrix([
        [5, 4, 0, 1],
        [0, 3, 4, 5],
        [1, 0, 5, 4]
    ])
    train, test = train_test_split(interaction_matrix, test_ratio=0.5)
    
    # Assert matrix shape matches original
    assert train.shape == interaction_matrix.shape
    assert test.shape == interaction_matrix.shape
    
    # Assert original values are retained in either train or test
    reconstructed = train + test
    assert np.allclose(reconstructed.toarray(), interaction_matrix.toarray())


def test_filter_popular_items(sample_reviews_df):
    cleaned = clean_reviews(sample_reviews_df)
    # After cleaning: I1 has 2 reviews (U1, U2), I2 has 2 (U1, U2).
    filtered = filter_popular_items(cleaned, min_reviews=2)
    assert set(filtered["item_id"].unique()) == {"I1", "I2"}
    # Raising the threshold above every item's count empties the frame.
    assert filter_popular_items(cleaned, min_reviews=5).empty
