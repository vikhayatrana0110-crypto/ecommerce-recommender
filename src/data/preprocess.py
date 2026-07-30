import pandas as pd
from scipy.sparse import csr_matrix

def clean_reviews(df):
    """Filters out incomplete rows and zero/negative ratings."""
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    return df[df["rating"] > 0]

def filter_active_users(df, min_reviews=1):
    """Filters users with at least min_reviews reviews."""
    return df[df.groupby('user_id')['user_id'].transform('size') >= min_reviews]

def filter_popular_items(df, min_reviews=1):
    """Filters items with at least min_reviews reviews."""
    return df[df.groupby('item_id')['item_id'].transform('size') >= min_reviews]

def create_interaction_matrix(df):
    """Creates a sparse interaction matrix from a DataFrame using vectorized mapping."""
    user_map = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    item_map = {iid: idx for idx, iid in enumerate(df["item_id"].unique())}
    
    # Vectorized mapping of IDs to matrix offsets
    row_indices = df["user_id"].map(user_map).values
    col_indices = df["item_id"].map(item_map).values
    ratings = df["rating"].values
    
    # Fast scipy sparse matrix creation
    matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(len(user_map), len(item_map)))
    return matrix, user_map, item_map
