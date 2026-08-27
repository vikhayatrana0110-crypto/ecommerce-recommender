import numpy as np
from scipy.sparse import csr_matrix

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def clean_reviews(df):
    """Filters out incomplete rows and zero/negative ratings."""
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    return df[df["rating"] > 0]


def dedupe_interactions(df):
    """Collapses repeat reviews of the same item by the same user.

    Without this, `csr_matrix` silently sums duplicate (user, item) pairs and
    inflates them into confidence values no rating scale ever produced.
    """
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(subset=["user_id", "item_id"], keep="last")


def filter_active_users(df, min_reviews=1):
    """Filters users with at least min_reviews reviews."""
    return df[df.groupby("user_id")["user_id"].transform("size") >= min_reviews]


def filter_popular_items(df, min_reviews=1):
    """Filters items with at least min_reviews reviews."""
    return df[df.groupby("item_id")["item_id"].transform("size") >= min_reviews]


def iterative_filter(df, min_reviews_user=5, min_reviews_item=5, max_passes=10):
    """Applies the user and item filters until the surviving set is stable.

    A single pass is not enough: dropping unpopular items pushes users back
    below the user threshold, which is what left the original pipeline at ~1.55
    interactions per user. Repeating to a fixed point yields the k-core.
    """
    for pass_no in range(1, max_passes + 1):
        before = len(df)
        df = filter_active_users(df, min_reviews_user)
        df = filter_popular_items(df, min_reviews_item)
        if len(df) == before:
            logger.info("k-core stable after %d pass(es): %s rows", pass_no, f"{len(df):,}")
            return df
        logger.debug("pass %d: %s -> %s rows", pass_no, f"{before:,}", f"{len(df):,}")
        if df.empty:
            logger.warning("Filtering removed every interaction; loosen the thresholds")
            return df

    logger.warning("k-core did not stabilise within %d passes", max_passes)
    return df


def binarize_ratings(df, threshold=4.0):
    """Keeps ratings at or above `threshold` and maps them to implicit 1.0."""
    df = df[df["rating"] >= threshold].copy()
    df["rating"] = 1.0
    return df


def describe(df, label=""):
    """Returns density statistics for logging before/after filtering."""
    users, items = df["user_id"].nunique(), df["item_id"].nunique()
    n = len(df)
    stats = {
        "label": label,
        "interactions": n,
        "users": users,
        "items": items,
        "density": n / (users * items) if users and items else 0.0,
        "per_user": n / users if users else 0.0,
    }
    logger.info(
        "%s: %s interactions | %s users x %s items | density %.4f | %.2f per user",
        label or "stats", f"{n:,}", f"{users:,}", f"{items:,}",
        stats["density"], stats["per_user"],
    )
    return stats


def create_interaction_matrix(df, user_map=None, item_map=None):
    """Builds a sparse interaction matrix using vectorized index mapping.

    Passing `user_map` / `item_map` rebuilds the matrix in an existing index
    space — required when serving a trained model, whose factors are only
    meaningful against the mapping it was trained with.
    """
    if user_map is None:
        user_map = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    if item_map is None:
        item_map = {iid: idx for idx, iid in enumerate(df["item_id"].unique())}

    rows = df["user_id"].map(user_map)
    cols = df["item_id"].map(item_map)

    # Drop interactions referencing IDs outside the supplied mapping.
    known = rows.notna() & cols.notna()
    if not known.all():
        logger.debug("Dropping %d interactions outside the supplied mapping", (~known).sum())

    ratings = df["rating"].to_numpy(dtype=np.float64)[known.to_numpy()]
    rows = rows[known].to_numpy(dtype=np.int64)
    cols = cols[known].to_numpy(dtype=np.int64)

    matrix = csr_matrix(
        (ratings, (rows, cols)), shape=(len(user_map), len(item_map))
    )
    return matrix, user_map, item_map
