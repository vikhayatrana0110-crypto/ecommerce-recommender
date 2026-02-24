import pandas as pd
from scipy.sparse import lil_matrix

def clean_reviews(df):
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df = df[df["rating"] > 0]
    return df



def filter_active_users(df, min_reviews=1):
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_reviews].index
    return df[df["user_id"].isin(active_users)]


def create_interaction_matrix(df):

    # Create ID mappings
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    num_users = len(user_map)
    num_items = len(item_map)

    matrix = lil_matrix((num_users, num_items))

    for _, row in df.iterrows():
        user_idx = user_map[row["user_id"]]
        item_idx = item_map[row["item_id"]]
        matrix[user_idx, item_idx] = row["rating"]  

    return matrix.tocsr(), user_map, item_map


def filter_popular_items(df, min_reviews=1):
    item_counts = df["item_id"].value_counts()
    popular_items = item_counts[item_counts >= min_reviews].index
    return df[df["item_id"].isin(popular_items)]
