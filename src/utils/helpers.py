import logging
import pickle
from pathlib import Path

import yaml

# Anchor every relative path to the project root so the pipeline works no matter
# which directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"

_LOG_CONFIGURED = False


def resolve_path(path):
    """Resolves a possibly-relative path against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads configuration parameters from a YAML file."""
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_pickle(obj, file_path):
    """Saves a python object as a pickle file, creating parent directories."""
    file_path = resolve_path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    """Loads a python object from a pickle file."""
    with open(resolve_path(file_path), "rb") as f:
        return pickle.load(f)


def get_logger(name, level="INFO"):
    """Returns a module logger, configuring root handlers once."""
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, str(level).upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def plot_metrics(results, output_path="reports/metrics.png"):
    """Renders a grouped bar chart comparing models across ranking metrics.

    `results` maps model name -> {metric name: score}.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(results)
    metrics = list(results[models[0]]) if models else []
    if not metrics:
        return None

    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3, 4.5))
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, [results[model][m] for m in metrics], width, label=model)
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)

    ax.set_xticks(x, metrics)
    ax.set_ylabel("Score")
    ax.set_title("Ranking performance by model")
    ax.legend()
    ax.margins(y=0.15)
    fig.tight_layout()

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
