# E-Commerce Recommendation System
### Implicit ALS collaborative filtering on the Amazon Electronics 2023 dataset, with a Streamlit dashboard

An end-to-end implicit-feedback recommender: streaming ingestion of multi-gigabyte
archives, k-core filtering, ALS matrix factorization, a content-based cold-start
path, four ranking metrics, and an interactive dashboard.

---

## Results

Temporal leave-last-out split, 94,249 evaluated users, k = 10. **ALS beats the
popularity baseline on every metric.**

| Model | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| **ALS** | **0.0024** | **0.0243** | **0.0130** | **0.0096** |
| Popularity | 0.0021 | 0.0208 | 0.0098 | 0.0066 |
| Hybrid | 0.0024 | 0.0243 | 0.0130 | 0.0096 |

ALS improves on the baseline by **+17% Recall@10, +33% NDCG@10, +45% MAP@10**.
The gap widens as the metric becomes more rank-sensitive, which is what you
expect when personalization is doing real work: ALS is not just retrieving more
relevant items, it is placing them higher.

Under leave-last-out every evaluated user has training history by construction,
so **Hybrid is identical to ALS here by design** — its cold-start path only
activates for users absent from training, which this protocol never produces.
It is exercised in the dashboard, not in this table.

> An earlier version of this project concluded the opposite — that popularity
> beat ALS "due to data sparsity". That result was an artifact of the pipeline,
> not a finding. See [Analysis](#analysis) for what was actually wrong.

![Ranking performance by model](reports/metrics.png)

---

## Dataset

Amazon Reviews 2023 — Electronics category (reviews 6.0 GB, metadata 1.2 GB, both
gzip-compressed JSONL, not tracked by git).

| Stage | Interactions | Users | Items | Per user |
|---|---|---|---|---|
| Raw (3M streamed) | 2,985,713 | 600,116 | 497,236 | 4.98 |
| After binarization (rating ≥ 4) | 2,359,187 | — | — | — |
| **After 5-core filtering** | **1,003,531** | **94,249** | **51,064** | **10.65** |

Final matrix: 94,249 × 51,064, density 0.0209%. Product titles resolve for
36,326 of 51,064 items (see [Data integrity](#data-integrity)).

### Pipeline
1. Stream reviews line-by-line, keeping only `user_id`, `parent_asin`, `rating`, `timestamp`
2. Drop incomplete rows and non-positive ratings
3. Collapse repeat reviews of the same item by the same user (keep most recent)
4. Binarize: `rating >= 4` becomes an implicit positive
5. **Iterative 5-core filtering** — repeat the user and item filters to a fixed point
6. Build the sparse CSR interaction matrix by vectorized index mapping
7. Temporal leave-last-out split

---

## Setup

Requires **Python 3.12** on macOS (Apple Silicon). `implicit` needs OpenMP:

```bash
brew install libomp
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Getting the data

The raw archives are gitignored because of their size, so a fresh clone needs
them downloaded once. They come from the
[Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/)
(McAuley Lab, UCSD), hosted on Hugging Face:

```bash
mkdir -p data/raw

# Reviews — 22.6 GB uncompressed
curl -L -C - -o data/raw/Electronics.jsonl \
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl

# Product metadata — 5.2 GB uncompressed
curl -L -C - -o data/raw/meta_Electronics.jsonl \
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Electronics.jsonl
```

These are large files on a slow endpoint — expect hours, not minutes. The server
supports range requests, and `curl -C -` resumes, so rerun the same command if a
transfer drops rather than starting over.

Then point [`src/config/config.yaml`](src/config/config.yaml) at whichever files
you have:

```yaml
data:
  reviews_path: "data/raw/Electronics.jsonl"
  metadata_path: "data/raw/meta_Electronics.jsonl"
```

Both plain `.jsonl` and gzip-compressed `.jsonl.gz` are accepted, so there is no
need to compress the downloads. The loaders stream line by line and stop at
`data.max_records` (3M by default), so the reviews file is never read in full —
only the metadata file is scanned deeply, to resolve product titles.

---

## Running

```bash
python main.py            # train, evaluate, write artifacts to data/
streamlit run app.py      # interactive dashboard
pytest                    # 81 tests
```

`main.py` takes roughly **1 min 45 s** on a cold cache (streaming and filtering
3M reviews) and **~57 s** thereafter, producing identical metrics either way —
the filtered interactions and resolved metadata are cached as
Parquet under `data/processed/`, keyed by a hash of the config values that
produced them, so editing `config.yaml` invalidates the cache automatically.

Everything resolves paths against the project root, so the commands work from
any working directory.

### Dashboard
- Recommendations for any trained user, with their purchase history
- **Cold start**: unknown user IDs fall back to the content/popularity path
- **Find similar products**: content-based lookalikes from titles and categories

---

## Configuration

All behaviour is driven by [`src/config/config.yaml`](src/config/config.yaml):

| Group | Controls |
|---|---|
| `data` | source paths, `max_records`, cache directory, metadata line budget |
| `filtering` | k-core thresholds, `max_filter_passes`, binarization threshold |
| `split` | `temporal` or `random`, test ratio |
| `als` | factors, regularization, iterations, confidence `alpha` |
| `eval` | `k`, optional user cap |
| `seed` | one seed for splitting, sampling and ALS initialisation |

---

## Architecture

```text
ecommerce-recommender/
├── src/
│   ├── config/config.yaml      # all hyperparameters and paths
│   ├── data/
│   │   ├── load_data.py        # streaming gzip JSONL readers
│   │   ├── preprocess.py       # dedupe, k-core filtering, matrix building
│   │   └── cache.py            # config-keyed Parquet cache
│   ├── models/
│   │   ├── als.py              # implicit ALS wrapper
│   │   ├── popularity.py       # baseline + cold-start floor
│   │   └── content.py          # TF-IDF item-item similarity
│   ├── eval/
│   │   ├── metrics.py          # Precision, Recall, NDCG, MAP
│   │   ├── split.py            # random and temporal splits
│   │   └── evaluate.py         # batched evaluation
│   ├── pipeline.py             # shared data preparation
│   └── utils/helpers.py        # config, paths, logging, plotting
├── tests/                      # 81 tests
├── app.py                      # Streamlit dashboard
└── main.py                     # train + evaluate + persist
```

---

## Analysis

The original pipeline reported ~1.55 interactions per user and concluded that
sparsity made ALS unviable. Four defects produced that number, and fixing them
reversed the result:

**1. Single-pass filtering.** Users were filtered, then items — but dropping
unpopular items pushes users back below the user threshold, and nothing checked
again. Repeating both filters to a fixed point (7 passes here) yields a genuine
5-core: **10.65 interactions per user instead of 1.55**. This was by far the
largest effect.

**2. Ratings used directly as confidence.** Star ratings were fed to ALS as
interaction strengths, so a 1-star review — evidence of *dislike* — became a
positive signal five times weaker than a 5-star one, rather than a negative.
Binarizing at `rating >= 4` states the implicit-feedback assumption correctly.

**3. A random split on temporally ordered data.** Holding out random items lets
the model train on a user's future to predict their past. Leave-last-out by
timestamp measures what a deployed system actually faces.

**4. Incomparable evaluation.** ALS was scored on the first 2,000 users while
popularity was scored on all of them, so the two numbers were never comparable.
Both models now run against an identical user set.

Absolute numbers remain low, as they should: 51,064 candidate items and one
held-out item per user means random guessing scores ~0.0002 Recall@10. ALS
reaches 0.0243 — roughly **124× random**.

---

## Data integrity

The results above were produced from **truncated copies** of both archives —
`gzip -t` failed on each. This is a property of those particular downloads, not
of the dataset: files fetched with the commands in
[Getting the data](#getting-the-data) are intact.

The readers are built to survive it regardless. They detect a corrupt region and
stop cleanly at the damage instead of crashing, so the pipeline runs on whatever
prefix is readable:

| Archive | Recoverable | Failure |
|---|---|---|
| `Electronics.jsonl.gz` | 36.3M reviews (clean well past 4M) | `invalid block type` near the end |
| `meta_Electronics.jsonl.gz` | ~844,529 records | degrades into one repeating record, then hits EOF |

The reviews file was unaffected in practice, since the pipeline reads 3M records
from a clean prefix. The metadata damage is why only 36,326 of 51,064 items
resolve a title; the rest render as "Unknown Product". **A clean metadata file
raises that coverage with no code change** — the reviews file does not need
re-downloading for the default 3M-record configuration.

---

## Testing

```bash
pytest
```

81 tests covering ranking metrics against hand-computed values, split
determinism and leakage, k-core convergence, duplicate-interaction handling,
index-space alignment between training and serving, loader behaviour on
truncated and malformed archives, and model round-trips.
