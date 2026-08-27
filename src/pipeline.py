"""Shared data pipeline used by both the training script and the dashboard."""
from scipy.sparse import load_npz, save_npz

from src.data.cache import cached_frame
from src.data.load_data import load_metadata, load_reviews
from src.data.preprocess import (
    binarize_ratings,
    clean_reviews,
    create_interaction_matrix,
    dedupe_interactions,
    describe,
    iterative_filter,
)
from src.utils.helpers import get_logger, resolve_path

logger = get_logger(__name__)


def build_interactions(cfg, use_cache=True):
    """Loads, cleans and filters reviews, caching the result as Parquet.

    The cache key covers every config value that changes the output, so editing
    config.yaml invalidates it automatically.
    """
    data_cfg, filt_cfg = cfg["data"], cfg["filtering"]

    key_parts = [
        data_cfg["reviews_path"], data_cfg["max_records"],
        filt_cfg["min_reviews_user"], filt_cfg["min_reviews_item"],
        filt_cfg["max_filter_passes"], filt_cfg["binarize"],
        filt_cfg.get("positive_threshold"),
    ]

    def builder():
        reviews = load_reviews(
            data_cfg["reviews_path"],
            max_records=data_cfg["max_records"],
            chunk_size=data_cfg.get("chunk_size", 250000),
        )
        reviews = clean_reviews(reviews)
        reviews = dedupe_interactions(reviews)
        describe(reviews, "before filtering")

        if filt_cfg.get("binarize", False):
            reviews = binarize_ratings(reviews, filt_cfg.get("positive_threshold", 4.0))
            logger.info("Binarized to %s positive interactions", f"{len(reviews):,}")

        reviews = iterative_filter(
            reviews,
            min_reviews_user=filt_cfg["min_reviews_user"],
            min_reviews_item=filt_cfg["min_reviews_item"],
            max_passes=filt_cfg["max_filter_passes"],
        )
        describe(reviews, "after filtering")
        return reviews.reset_index(drop=True)

    return cached_frame(
        "interactions", builder, data_cfg["cache_dir"], key_parts, enabled=use_cache
    )


def build_metadata(cfg, item_ids, use_cache=True):
    """Loads titles/categories for the active items only, cached as Parquet."""
    data_cfg = cfg["data"]
    item_ids = sorted(map(str, item_ids))
    key_parts = [data_cfg["metadata_path"], len(item_ids), item_ids[:50], item_ids[-50:]]

    def builder():
        return load_metadata(
            data_cfg["metadata_path"],
            filter_items=item_ids,
            max_lines=data_cfg.get("metadata_max_lines"),
        )

    df = cached_frame(
        "metadata", builder, data_cfg["cache_dir"], key_parts, enabled=use_cache
    )
    # categories arrives as a list column; parquet round-trips it as ndarray.
    return df


def save_matrix(matrix, path):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, matrix.tocsr())


def load_matrix(path):
    return load_npz(resolve_path(path))


def prepare(cfg, use_cache=True):
    """Full pipeline: interactions frame, matrix, mappings and metadata."""
    reviews = build_interactions(cfg, use_cache=use_cache)
    matrix, user_map, item_map = create_interaction_matrix(reviews)
    metadata = build_metadata(cfg, item_map.keys(), use_cache=use_cache)
    return reviews, matrix, user_map, item_map, metadata


def title_lookup(metadata):
    """item_id -> product title, for human-readable output."""
    if metadata.empty:
        return {}
    return dict(zip(metadata["item_id"], metadata["title"]))
