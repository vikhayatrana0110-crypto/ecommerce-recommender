# Project Context: E-commerce Recommender System

An implicit-feedback collaborative filtering recommender for the Amazon Reviews
2023 Electronics dataset. Trains ALS matrix factorization, benchmarks it against
a popularity baseline and a hybrid cold-start model, and serves results through
a Streamlit dashboard.

## Entry points

- **`main.py`** — trains, evaluates and persists. Also defines `HybridRecommender`
  (ALS for warm users, content/popularity for cold ones).
- **`app.py`** — Streamlit dashboard. Loads persisted artifacts; never retrains.
- **`src/pipeline.py`** — shared data preparation used by both, so the app and the
  trainer cannot drift apart.

## Modules

| Path | Responsibility |
|---|---|
| `src/data/load_data.py` | Streaming gzip JSONL readers; tolerate truncated archives |
| `src/data/preprocess.py` | Dedupe, binarize, iterative k-core filter, CSR matrix building |
| `src/data/cache.py` | Config-keyed Parquet cache |
| `src/models/als.py` | `implicit` ALS wrapper: train, batch-recommend, persist |
| `src/models/popularity.py` | Popularity baseline and cold-start floor |
| `src/models/content.py` | TF-IDF item-item similarity; cold-start profiles |
| `src/eval/metrics.py` | Precision@K, Recall@K, NDCG@K, MAP@K |
| `src/eval/split.py` | Seeded random split and temporal leave-last-out |
| `src/eval/evaluate.py` | Batched evaluation over a shared user set |
| `src/utils/helpers.py` | Config loading, path resolution, logging, plotting |

## Conventions

- **Paths**: every relative path resolves against `PROJECT_ROOT` via
  `resolve_path()` in `src/utils/helpers.py`. Nothing depends on the CWD.
- **Config**: no hyperparameter is hardcoded; everything lives in
  `src/config/config.yaml`.
- **Determinism**: one `seed` drives the split, user sampling and ALS
  initialisation. Two runs produce byte-identical metrics.
- **Index space**: the ALS model's factors are only meaningful against the
  `user_map` / `item_map` it was trained with. Anything rebuilding an
  interaction matrix for serving must pass those maps into
  `create_interaction_matrix`.
- **Caching**: cache keys hash the config values that produced the frame, so
  changing `config.yaml` invalidates automatically.

## Core technologies

Python 3.12 · `implicit` (ALS) · SciPy sparse · pandas / NumPy ·
scikit-learn (TF-IDF) · PyArrow (Parquet cache) · Streamlit · pytest

## Known constraints

- Both raw archives are truncated downloads; see README > Data integrity.
  Metadata resolves titles for 36,311 of 51,064 items.
- macOS is the only supported platform; `implicit` requires `libomp`.
