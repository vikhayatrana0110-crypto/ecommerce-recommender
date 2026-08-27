"""Trains the ALS recommender, benchmarks it, and persists the artifacts."""
import numpy as np

from src.eval.evaluate import eligible_users, evaluate_model, format_results_table
from src.eval.split import build_split, random_split
from src.models.als import ALSRecommender
from src.models.content import ContentRecommender, cold_start_recommend
from src.models.popularity import PopularityRecommender
from src.pipeline import prepare, save_matrix, title_lookup
from src.utils.helpers import get_logger, load_config, plot_metrics, resolve_path, save_pickle

logger = get_logger(__name__)

# Kept at module level: tests/test_recommender.py imports this name.
train_test_split = random_split


class HybridRecommender:
    """ALS for known users, content/popularity cold start for cold ones.

    A user with no training history has no ALS embedding worth trusting, so
    routing them to the content profile is strictly better than serving noise.
    """

    def __init__(self, als, content, popularity):
        self.als = als
        self.content = content
        self.popularity = popularity
        self._cold_users = 0

    def recommend_batch(self, user_ids, train_matrix, n=10):
        user_ids = np.asarray(user_ids)
        # One vectorised pass instead of slicing a sparse row per user.
        history = np.diff(train_matrix.indptr)[user_ids]
        warm_mask = history > 0

        results = [None] * len(user_ids)

        warm = user_ids[warm_mask]
        if len(warm):
            warm_recs = self.als.recommend_batch(warm, train_matrix, n=n)
            for pos, rec in zip(np.flatnonzero(warm_mask), warm_recs):
                results[pos] = rec

        for pos in np.flatnonzero(~warm_mask):
            user = user_ids[pos]
            results[pos] = cold_start_recommend(
                self.content, self.popularity, train_matrix[user].indices, n=n
            )

        return results

    @property
    def cold_user_count(self):
        return self._cold_users

def main():
    cfg = load_config()
    logger_level = cfg.get("logging", {}).get("level", "INFO")
    logger.setLevel(logger_level)
    seed = cfg.get("seed")
    k = cfg["eval"]["k"]

    logger.info("Preparing data...")
    reviews, matrix, user_map, item_map, metadata = prepare(cfg)

    cells = matrix.shape[0] * matrix.shape[1]
    if cells == 0:
        raise SystemExit(
            "Filtering left no interactions. Lower filtering.min_reviews_user / "
            "min_reviews_item, or raise data.max_records, in src/config/config.yaml."
        )
    logger.info(
        "Interaction matrix: %s users x %s items, %s non-zeros (density %.4f)",
        f"{matrix.shape[0]:,}", f"{matrix.shape[1]:,}", f"{matrix.nnz:,}",
        matrix.nnz / cells,
    )

    strategy = cfg["split"]["strategy"]
    logger.info("Splitting (%s)...", strategy)
    train_matrix, test_matrix = build_split(reviews, matrix, user_map, item_map, cfg)

    # Models
    als = ALSRecommender(
        factors=cfg["als"]["factors"],
        regularization=cfg["als"]["regularization"],
        iterations=cfg["als"]["iterations"],
        alpha=cfg["als"]["alpha"],
        seed=seed,
    ).fit(train_matrix)

    popularity = PopularityRecommender().fit(train_matrix)
    content = ContentRecommender().fit(metadata, item_map)
    hybrid = HybridRecommender(als, content, popularity)

    # Persist artifacts
    als.save(cfg["output"]["model_path"])
    save_pickle(user_map, cfg["output"]["user_map_path"])
    save_pickle(item_map, cfg["output"]["item_map_path"])
    save_matrix(matrix, cfg["output"]["matrix_path"])
    logger.info("Artifacts saved to %s", resolve_path(cfg["output"]["model_path"]).parent)

    # Evaluation — every model scored on the identical user set.
    users = eligible_users(test_matrix, cfg["eval"].get("max_users"), seed=seed)
    logger.info("Evaluating %s users with held-out items", f"{len(users):,}")

    cold = int((np.diff(train_matrix.indptr)[users] == 0).sum())
    logger.info(
        "%s of %s evaluated users have no training history (cold)",
        f"{cold:,}", f"{len(users):,}",
    )

    results = {}
    for name, model in (
        ("ALS", als), ("Popularity", popularity), ("Hybrid", hybrid),
    ):
        scores = evaluate_model(name, model, users, train_matrix, test_matrix, k=k)
        if scores:
            results[name] = scores

    if cold == 0:
        logger.info(
            "Hybrid matches ALS here by construction: leave-last-out never "
            "produces a cold user. Its cold-start path is exercised by the app."
        )

    print(f"\nRanking performance ({strategy} split, k={k}, {len(users):,} users)\n")
    print(format_results_table(results))

    chart = plot_metrics(results, resolve_path(cfg["output"]["reports_dir"]) / "metrics.png")
    if chart:
        logger.info("Metrics chart written to %s", chart)

    # Demo: readable recommendations for one user.
    titles = title_lookup(metadata)
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}

    if len(users):
        demo = int(users[0])
        recs = als.recommend_batch([demo], train_matrix, n=k)[0]
        print(f"\nSample recommendations for user {inv_user_map[demo]}:")
        for rank, item_idx in enumerate(recs, start=1):
            asin = inv_item_map[int(item_idx)]
            print(f"  {rank:2}. {titles.get(asin, 'Unknown Product')}  ({asin})")


if __name__ == "__main__":
    main()
