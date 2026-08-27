# Future Work

The gaps listed in the original version of this file are closed. What follows is
what a next iteration would tackle, in rough order of value.

## Closed

| Item | Resolution |
|---|---|
| Unused metadata; no content-based model | `src/models/content.py` — TF-IDF item-item similarity, cold-start profiles, "similar products" in the dashboard |
| Empty config; hardcoded hyperparameters | `src/config/config.yaml` drives every threshold, path and hyperparameter |
| Empty helpers module | `src/utils/helpers.py` — config, path resolution, pickling, logging, plotting |
| Inconsistent evaluation | All models scored on an identical user set via `src/eval/evaluate.py` |
| Missing NDCG / MAP | `src/eval/metrics.py`, verified against hand-computed values |
| Slow ingestion; `.iterrows()` | Line-streaming readers; vectorized index mapping; Parquet cache |
| No model persistence | ALS model, maps and matrix persisted; the app never retrains |
| No application interface | Streamlit dashboard with cold start and similar-products |
| No documentation or tests | README rewritten from real runs; 78 tests |
| Non-deterministic results | One `seed`; two runs produce byte-identical metrics |
| Sparsity mis-diagnosis | Iterative k-core filtering: 1.55 → 10.65 interactions per user |

## Open

### 1. Re-download the raw archives
Both are truncated (see README > Data integrity). Only 36,311 of 51,064 items
resolve a title, so the dashboard shows "Unknown Product" for ~29% of results
and the content model has nothing to work with for those items. **This is the
single highest-value fix and needs no code change.**

### 2. Hyperparameter search
`factors`, `regularization` and `alpha` are reasonable defaults, never tuned. A
sweep with a proper validation split would likely move the ALS numbers
meaningfully.

### 3. Scale past 3M reviews
The clean prefix holds ~36M reviews; the pipeline currently reads 3M. Going
further needs out-of-core matrix construction — the interactions frame is
already the memory ceiling.

### 4. Evaluate the cold-start path
Hybrid is identical to ALS under leave-last-out because that protocol never
produces a cold user. Measuring it needs a held-out-user protocol where entire
users are withheld from training.

### 5. Richer models
BPR or LightFM (which folds item features directly into factorization) would be
a natural comparison, and would make the content signal part of the model rather
than only a fallback.

### 6. Serving interface
Deliberately out of scope for this iteration: no CLI or REST API. The Streamlit
dashboard is the only interface. A FastAPI service loading the pickled artifacts
would be straightforward if one is ever needed.

### 7. Cross-platform support
macOS only. The path handling is already CWD- and separator-independent, so a
Windows/Linux pass would mostly be dependency pinning and CI — but nothing is
verified off macOS today.
