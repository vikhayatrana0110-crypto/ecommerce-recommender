from src.data.load_data import load_reviews, load_metadata
from src.data.preprocess import clean_reviews, filter_active_users, filter_popular_items
from src.data.preprocess import create_interaction_matrix
from scipy.sparse import csr_matrix
import numpy as np
from implicit.als import AlternatingLeastSquares



REVIEWS_PATH = "data/raw/Electronics.jsonl.gz"
META_PATH = "data/raw/meta_Electronics.jsonl.gz"

def train_test_split(interaction_matrix, test_ratio=0.2):

    train = interaction_matrix.copy().tolil()
    test = csr_matrix(interaction_matrix.shape)

    num_users = interaction_matrix.shape[0]

    for user in range(num_users):

        start = interaction_matrix.indptr[user]
        end = interaction_matrix.indptr[user + 1]
        item_indices = interaction_matrix.indices[start:end]

        if len(item_indices) < 2:
            continue

        test_size = max(1, int(len(item_indices) * test_ratio))
        test_items = np.random.choice(item_indices, size=test_size, replace=False)

        for item in test_items:
            train[user, item] = 0

        test[user, test_items] = interaction_matrix[user, test_items]

    return train.tocsr(), test


def recommend_items(user_id, predictions, train_matrix, N=10):
    user_predictions = predictions[user_id].copy()
    known_items = train_matrix[user_id].indices
    user_predictions[known_items] = -np.inf
    return np.argsort(-user_predictions)[:N]


def precision_at_k(predictions, test_matrix, train_matrix, k=10):
    num_users = test_matrix.shape[0]
    precisions = []

    for user in range(num_users):
        #skip users with no test items
        test_items = test_matrix[user].indices
        if len(test_items) == 0:
            continue
        #get user prediction
        user_predictions = predictions[user].copy()
        #remove items seen in training
        train_items = train_matrix[user].indices
        user_predictions[train_items] = -np.inf
        #top-k recommendations
        top_k_items = np.argsort(-user_predictions)[:k]

        #compute hits
        hits = len(set(top_k_items) & set(test_items))

        precisions.append(hits / k)
    if len(precisions) == 0:
        return 0.0
    return np.mean(precisions)

def recall_at_k(predictions, test_matrix, train_matrix, k=10):

    num_users = test_matrix.shape[0]
    recalls = []

    for user in range(num_users):

        test_items = test_matrix[user].indices
        if len(test_items) == 0:
            continue

        user_predictions = predictions[user].copy()
        train_items = train_matrix[user].indices
        user_predictions[train_items] = -np.inf

        top_k_items = np.argsort(-user_predictions)[:k]

        hits = len(set(top_k_items) & set(test_items))

        recalls.append(hits / len(test_items))

    if len(recalls) == 0:
        return 0.0

    return np.mean(recalls)

def popularity_baseline(train_matrix, k=10):

    # Count item popularity
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()

    # Get top-k globally popular items
    top_k_items = np.argsort(-item_popularity)[:k]

    return top_k_items


def evaluate_popularity(test_matrix, train_matrix, k=10):

    top_k_items = popularity_baseline(train_matrix, k)

    num_users = test_matrix.shape[0]
    precisions = []
    recalls = []

    for user in range(num_users):

        test_items = test_matrix[user].indices
        if len(test_items) == 0:
            continue

        hits = len(set(top_k_items) & set(test_items))

        precisions.append(hits / k)
        recalls.append(hits / len(test_items))

    return np.mean(precisions), np.mean(recalls)
  
def evaluate_als(model, train_matrix, test_matrix,k=20, max_users=2000):

    precision_sum = 0
    recall_sum = 0
    evaluated_users = 0
    

    num_users = train_matrix.shape[0]

    for user_id in range(min(num_users, max_users)):

        test_items = test_matrix[user_id].indices
        if len(test_items) == 0:
            continue
        recommended, _ = model.recommend(
            user_id ,
            user_items = train_matrix[user_id],
            N=k,
            filter_already_liked_items=True
        )

        hits = len(set(recommended) & set(test_items))

        precision_sum += hits / k
        recall_sum += hits / len(test_items)
        evaluated_users += 1

    if evaluated_users == 0:
        return 0, 0

    return precision_sum / evaluated_users, recall_sum / evaluated_users


def main():
    print("Loading reviews...")
    reviews = load_reviews(REVIEWS_PATH, max_records=50000)

    print("Loading metadata...")
    metadata = load_metadata(META_PATH, max_records=50000)
    print(reviews.head())

    print(f"Initial reviews: {len(reviews)}")
    print(f"Initial metadata: {len(metadata)}")

    # Step 1: Clean reviews
    reviews = clean_reviews(reviews)
    print(f"After clean_reviews: {len(reviews)}")

    # Step 2: Filter active users
    reviews = filter_active_users(reviews, min_reviews=10)
    print(f"After filter_active_users: {len(reviews)}")

    # Step 3: Filter popular items
    reviews = filter_popular_items(reviews, min_reviews=10)
    print(f"After filter_popular_items: {len(reviews)}")


    # interaction matrix
    interaction_matrix , user_map , item_map = create_interaction_matrix(reviews)
    print("interaction matrix shape:", interaction_matrix.shape)
    print("Number of non-zero interactions:", interaction_matrix.nnz)

    num_users, num_items = interaction_matrix.shape
    density = interaction_matrix.nnz / (num_users * num_items)
    print("Matrix density:", density)

    train_matrix, test_matrix = train_test_split(interaction_matrix)
    print("Train/Test split complete")
    

    
    alpha = 40  # confidence scaling factor

    # Confidence matrix
    confidence_matrix = (train_matrix* alpha).astype("double")

    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=25,
        use_gpu=False
    )
   
    # implicit expects item-user matrix
    model.fit(confidence_matrix)

    print("ALS training complete")


    print("Model trained successfully.")
    
    print("Train matrix shape:", train_matrix.shape)
    print("Confidence matrix shape:", confidence_matrix.shape)
    print("Model user factors shape:", model.user_factors.shape)
    print("Model item factors shape:", model.item_factors.shape)

    als_precision, als_recall = evaluate_als(
        model,
        train_matrix,
        test_matrix,
        k=10
    )

    print(f"ALS Precision@10: {als_precision:.4f}")
    print(f"ALS Recall@10: {als_recall:.4f}")

    
    pop_precision, pop_recall = evaluate_popularity(test_matrix, train_matrix, k=10)
    print(f"Popularity Precision@10: {pop_precision:.4f}")
    print(f"Popularity Recall@10: {pop_recall:.4f}")
    
    print("Users:", train_matrix.shape[0])
    print("Items:", train_matrix.shape[1])
    print("Density:", train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]))
    

    import numpy as np

    user_interactions = train_matrix.getnnz(axis=1)

    print("Average interactions per user:", user_interactions.mean())
    print("Median interactions per user:", np.median(user_interactions))
    print("Min interactions:", user_interactions.min())
    print("Max interactions:", user_interactions.max())
if __name__ == "__main__":
    main()