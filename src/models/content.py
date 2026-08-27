import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _as_text(value):
    """Flattens the metadata fields, which mix strings, lists and NaN."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


class ContentRecommender:
    """Item-item similarity over TF-IDF of product title + categories.

    This is what lets the system answer for a user the ALS model has never
    seen: given anything they interacted with, recommend lookalike products
    instead of falling straight back to global popularity.
    """

    def __init__(self, max_features=20000, min_df=1):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words="english",
        )
        self.item_map = None
        self.index_to_item = None
        self.vectors = None

    def fit(self, metadata, item_map):
        """Builds normalised TF-IDF vectors aligned to the ALS item index space."""
        self.item_map = item_map
        self.index_to_item = {v: k for k, v in item_map.items()}

        # Row i of the matrix must correspond to item index i in the model.
        text_by_item = {}
        for row in metadata.itertuples(index=False):
            text_by_item[row.item_id] = " ".join(
                (
                    _as_text(getattr(row, "title", "")),
                    _as_text(getattr(row, "categories", "")),
                    _as_text(getattr(row, "main_category", "")),
                )
            ).strip()

        corpus = [
            text_by_item.get(self.index_to_item.get(i, None), "")
            for i in range(len(item_map))
        ]
        covered = sum(1 for t in corpus if t)
        logger.info(
            "Content model: %d/%d items have metadata text", covered, len(corpus)
        )

        if covered == 0:
            logger.warning("No metadata text available; content model disabled")
            self.vectors = None
            return self

        self.vectors = normalize(self.vectorizer.fit_transform(corpus))
        return self

    @property
    def available(self):
        return self.vectors is not None

    def similar_items(self, item_index, n=10):
        """Top-n most similar items by cosine similarity, excluding itself."""
        if not self.available:
            return np.array([], dtype=int)
        scores = (self.vectors @ self.vectors[item_index].T).toarray().ravel()
        scores[item_index] = -np.inf
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        return top[np.argsort(-scores[top])]

    def recommend_from_profile(self, seen_indices, n=10):
        """Recommends against the centroid of everything the user has touched."""
        seen_indices = [i for i in seen_indices if 0 <= i < self.vectors.shape[0]] \
            if self.available else []
        if not seen_indices:
            return np.array([], dtype=int)

        # .mean on a sparse matrix yields np.matrix, which sklearn rejects.
        centroid = np.asarray(self.vectors[seen_indices].mean(axis=0))
        profile = normalize(centroid)
        scores = np.asarray(profile @ self.vectors.T).ravel()
        scores[seen_indices] = -np.inf
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        return top[np.argsort(-scores[top])]


def cold_start_recommend(content_model, popularity_model, seen_indices, n=10):
    """Content profile when we know anything about the user; popularity otherwise.

    Pads with popular items so the caller always receives n recommendations.
    """
    recs = []
    if content_model is not None and content_model.available and len(seen_indices) > 0:
        recs = list(content_model.recommend_from_profile(seen_indices, n=n))

    if len(recs) < n:
        exclude = set(recs) | set(seen_indices)
        filler = popularity_model.recommend(n=n - len(recs), exclude=exclude)
        recs.extend(filler)

    return np.array(recs[:n], dtype=int)
