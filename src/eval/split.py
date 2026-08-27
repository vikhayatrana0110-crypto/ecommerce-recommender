import numpy as np
from scipy.sparse import csr_matrix

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def random_split(interaction_matrix, test_ratio=0.2, seed=None):
    """Holds out a random fraction of each user's items.

    Users with a single interaction stay wholly in train — there is nothing to
    hold out without erasing their entire history. A seed makes the split (and
    therefore every downstream metric) reproducible.
    """
    rng = np.random.default_rng(seed)
    matrix = interaction_matrix.tocsr()

    train = matrix.copy().tolil()
    test = csr_matrix(matrix.shape).tolil()

    for user in range(matrix.shape[0]):
        start, end = matrix.indptr[user], matrix.indptr[user + 1]
        item_indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        if len(item_indices) < 2:
            continue

        test_size = max(1, int(len(item_indices) * test_ratio))
        chosen = rng.choice(len(item_indices), size=test_size, replace=False)
        for pos in chosen:
            item = item_indices[pos]
            train[user, item] = 0
            test[user, item] = values[pos]

    return train.tocsr(), test.tocsr()


def temporal_leave_last_out(df, user_map, item_map, n_holdout=1):
    """Splits chronologically: each user's most recent interaction(s) are the test set.

    This is the honest protocol for a recommender — a random split lets the
    model train on a user's future to predict their past, which flatters
    collaborative filtering and makes offline numbers unreproducible in serving.
    """
    if "timestamp" not in df.columns:
        raise KeyError("temporal split requires a 'timestamp' column")

    df = df.sort_values(["user_id", "timestamp"])
    rank_from_end = df.groupby("user_id").cumcount(ascending=False)
    sizes = df.groupby("user_id")["user_id"].transform("size")

    # Keep single-interaction users entirely in train.
    is_test = (rank_from_end < n_holdout) & (sizes > n_holdout)

    shape = (len(user_map), len(item_map))

    def to_matrix(subset):
        rows = subset["user_id"].map(user_map)
        cols = subset["item_id"].map(item_map)
        known = rows.notna() & cols.notna()
        mask = known.to_numpy()
        return csr_matrix(
            (
                subset["rating"].to_numpy(dtype=np.float64)[mask],
                (rows[known].to_numpy(dtype=np.int64), cols[known].to_numpy(dtype=np.int64)),
            ),
            shape=shape,
        )

    train, test = to_matrix(df[~is_test]), to_matrix(df[is_test])
    logger.info(
        "Temporal split: %s train / %s test interactions (%s users held out)",
        f"{train.nnz:,}", f"{test.nnz:,}", f"{int(is_test.sum()):,}",
    )
    return train, test


def build_split(df, interaction_matrix, user_map, item_map, cfg):
    """Dispatches to the split strategy named in config."""
    strategy = cfg.get("split", {}).get("strategy", "random")
    if strategy == "temporal":
        return temporal_leave_last_out(df, user_map, item_map)
    return random_split(
        interaction_matrix,
        test_ratio=cfg.get("split", {}).get("test_ratio", 0.2),
        seed=cfg.get("seed"),
    )
