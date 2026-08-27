import numpy as np
from implicit.als import AlternatingLeastSquares

from src.utils.helpers import get_logger, load_pickle, save_pickle

logger = get_logger(__name__)


class ALSRecommender:
    """Thin wrapper over implicit's ALS: train, batch-recommend, persist."""

    def __init__(self, factors=64, regularization=0.01, iterations=25, alpha=40, seed=None):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            # implicit >=0.7 applies the confidence scaling internally, so the
            # caller must not pre-multiply the matrix as well.
            alpha=alpha,
            random_state=seed,
            use_gpu=False,
        )

    def fit(self, train_matrix):
        logger.info(
            "Training ALS: %d factors, %d iterations",
            self.model.factors, self.model.iterations,
        )
        self.model.fit(train_matrix.astype(np.float32))
        return self

    def recommend_batch(self, user_ids, train_matrix, n=10, filter_seen=True):
        """One batched call — far faster than looping `recommend` per user."""
        user_ids = np.asarray(user_ids)
        ids, _ = self.model.recommend(
            user_ids,
            train_matrix[user_ids],
            N=n,
            filter_already_liked_items=filter_seen,
        )
        return [row for row in ids]

    def recommend(self, user_id, user_items, n=10, filter_seen=True):
        ids, _ = self.model.recommend(
            user_id, user_items, N=n, filter_already_liked_items=filter_seen
        )
        return ids

    def save(self, path):
        save_pickle(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.model = load_pickle(path)
        return obj
