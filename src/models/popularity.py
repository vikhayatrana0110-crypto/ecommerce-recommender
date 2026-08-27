import numpy as np


class PopularityRecommender:
    """Non-personalised baseline: rank items by total interaction weight.

    Also the cold-start floor — it needs no user history at all, so it can
    always answer when a personalised model cannot.
    """

    def __init__(self):
        self.item_scores = None
        self.ranking = None

    def fit(self, interaction_matrix):
        self.item_scores = np.asarray(interaction_matrix.sum(axis=0)).ravel()
        self.ranking = np.argsort(-self.item_scores)
        return self

    def recommend(self, n=10, exclude=None):
        """Returns the top-n most popular item indices, skipping `exclude`.

        Only the first n + |exclude| ranks can matter, so the scan stays bounded
        instead of walking the full catalogue for every user.
        """
        if self.ranking is None:
            raise RuntimeError("PopularityRecommender.fit must be called first")
        if exclude is None or len(exclude) == 0:
            return self.ranking[:n]

        exclude = set(int(i) for i in exclude)
        window = self.ranking[: n + len(exclude)]
        picks = [i for i in window if i not in exclude][:n]

        # Fall back to a full scan only if the window was somehow exhausted.
        if len(picks) < n:
            picks = [i for i in self.ranking if i not in exclude][:n]
        return np.array(picks, dtype=int)

    def recommend_batch(self, user_ids, train_matrix, n=10, filter_seen=True):
        """Top-n per user, optionally filtering each user's already-seen items."""
        if not filter_seen:
            top = self.ranking[:n]
            return [top for _ in user_ids]
        return [
            self.recommend(n=n, exclude=train_matrix[u].indices) for u in user_ids
        ]
