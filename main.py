import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from src.data.load_data import load_reviews, load_metadata
from src.data.preprocess import clean_reviews, filter_active_users, filter_popular_items, create_interaction_matrix
from src.utils.helpers import load_config, save_pickle, load_pickle

def train_test_split(interaction_matrix, test_ratio=0.2):
    """Splits interaction matrix into training and testing matrices by masking test items."""
    train = interaction_matrix.copy().tolil()
    test = csr_matrix(interaction_matrix.shape).tolil()
    
    for user in range(interaction_matrix.shape[0]):
        start, end = interaction_matrix.indptr[user], interaction_matrix.indptr[user + 1]
        item_indices = interaction_matrix.indices[start:end]
        
        if len(item_indices) >= 2:
            test_size = max(1, int(len(item_indices) * test_ratio))
            test_items = np.random.choice(item_indices, size=test_size, replace=False)
            train[user, test_items] = 0
            test[user, test_items] = interaction_matrix[user, test_items]
            
    return train.tocsr(), test.tocsr()

def evaluate_recommender(test_matrix, train_matrix, recommend_fn, k=10, max_users=2000):
    """Evaluates recall and precision metrics consistently across models."""
    precisions, recalls = [], []
    num_users = min(test_matrix.shape[0], max_users)
    
    for user_id in range(num_users):
        test_items = test_matrix[user_id].indices
        if len(test_items) == 0:
            continue
        
        # Call model-specific recommendation function
        recommended = recommend_fn(user_id, train_matrix[user_id])
        hits = len(set(recommended) & set(test_items))
        
        precisions.append(hits / k)
        recalls.append(hits / len(test_items))
        
    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0

def main():
    # 1. Initialization and configuration loading
    cfg = load_config()
    os.makedirs(os.path.dirname(cfg['output']['model_path']), exist_ok=True)
    
    print("Loading data...")
    reviews = load_reviews(cfg['data']['reviews_path'], cfg['data']['max_records'])
    
    # 2. Filtering and Preprocessing
    print("Preprocessing dataset...")
    reviews = clean_reviews(reviews)
    reviews = filter_active_users(reviews, cfg['filtering']['min_reviews_user'])
    reviews = filter_popular_items(reviews, cfg['filtering']['min_reviews_item'])
    
    # Dynamically resolve metadata only for active products in the interaction matrix
    print("Loading metadata...")
    active_item_ids = reviews['item_id'].unique()
    metadata = load_metadata(cfg['data']['metadata_path'], filter_items=active_item_ids)
    
    interaction_matrix, user_map, item_map = create_interaction_matrix(reviews)
    train_matrix, test_matrix = train_test_split(interaction_matrix)
    
    # 3. Model Training (implicit ALS)
    print("Training ALS model...")
    confidence_matrix = (train_matrix * cfg['als']['alpha']).astype("double")
    model = AlternatingLeastSquares(
        factors=cfg['als']['factors'],
        regularization=cfg['als']['regularization'],
        iterations=cfg['als']['iterations'],
        use_gpu=False
    )
    model.fit(confidence_matrix)
    
    # Save model and mappings
    save_pickle(model, cfg['output']['model_path'])
    save_pickle(user_map, cfg['output']['user_map_path'])
    save_pickle(item_map, cfg['output']['item_map_path'])
    print(f"Model saved successfully to {cfg['output']['model_path']}")
    
    # 4. Consistent Evaluation
    # ALS Recommendations fn
    def recommend_als(user_id, user_items):
        recs, _ = model.recommend(user_id, user_items=user_items, N=10, filter_already_liked_items=True)
        return recs
        
    # Popularity Baseline fn
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    top_k_popular = np.argsort(-item_popularity)[:10]
    def recommend_pop(user_id, user_items):
        return top_k_popular

    als_p, als_r = evaluate_recommender(test_matrix, train_matrix, recommend_als)
    pop_p, pop_r = evaluate_recommender(test_matrix, train_matrix, recommend_pop)
    
    print(f"\nALS Precision@10: {als_p:.4f} | Recall@10: {als_r:.4f}")
    print(f"Popularity Precision@10: {pop_p:.4f} | Recall@10: {pop_r:.4f}")
    
    # 5. Metadata Integration Showcase (Recommendation Demo)
    # Create product map from metadata DataFrame
    meta_dict = metadata.set_index('item_id')['title'].to_dict()
    # Reverse maps to translate indices back to IDs
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}
    
    demo_user = 0
    raw_user_id = inv_user_map[demo_user]
    recommended_indices = recommend_als(demo_user, train_matrix[demo_user])
    
    print(f"\nRecommendations for User: {raw_user_id}")
    for idx, item_idx in enumerate(recommended_indices):
        asin = inv_item_map[item_idx]
        title = meta_dict.get(asin, "Unknown Product")
        print(f"  {idx+1}. {title} (ASIN: {asin})")

if __name__ == "__main__":
    main()