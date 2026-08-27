import numpy as np

from src.eval.metrics import evaluate_rankings
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def eligible_users(test_matrix, max_users=None, seed=None):
    """Users with at least one held-out item.

    Every model is scored on this identical set. The original pipeline compared
    ALS on the first 2000 users against popularity on all of them, which made
    the two numbers incomparable.
    """
    users = np.where(np.diff(test_matrix.indptr) > 0)[0]
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    return users


def evaluate_model(name, recommender, user_ids, train_matrix, test_matrix, k=10):
    """Scores one recommender over a fixed user set with all four metrics."""
    if len(user_ids) == 0:
        logger.warning("%s: no users with held-out items to evaluate", name)
        return {}

    recommendations = recommender.recommend_batch(user_ids, train_matrix, n=k)
    relevants = [set(test_matrix[u].indices) for u in user_ids]

    results = evaluate_rankings(recommendations, relevants, k)
    logger.info(
        "%-12s %s", name, " | ".join(f"{m}: {v:.4f}" for m, v in results.items())
    )
    return results


def format_results_table(results_by_model):
    """Renders a Markdown table ready to paste into the README."""
    models = list(results_by_model)
    if not models:
        return ""
    metrics = list(results_by_model[models[0]])

    best = {
        m: max(models, key=lambda mo: results_by_model[mo].get(m, 0.0))
        for m in metrics
    }

    lines = [
        "| Model | " + " | ".join(metrics) + " |",
        "|" + "---|" * (len(metrics) + 1),
    ]
    for model in models:
        cells = []
        for m in metrics:
            value = f"{results_by_model[model].get(m, 0.0):.4f}"
            cells.append(f"**{value}**" if best[m] == model else value)
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
