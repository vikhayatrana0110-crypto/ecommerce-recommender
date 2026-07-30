# Incomplete Features and Shortcomings

Below is a detailed list of features left incomplete, architectural gaps, and general shortcomings identified in the current state of the E-commerce Recommender project.

---

## 1. Unused Metadata & Lack of Content-Based Recommenders
- **Incomplete**: Although `load_metadata` exists in `src/data/load_data.py` and is called in `main.py`, the metadata DataFrame is never used.
- **Shortcoming**: 
  - The recommendations are solely collaborative (using ratings/interactions).
  - There is no fallback or hybrid model using metadata titles and categories to solve the **cold-start problem** for new users or items.
  - The final outputs cannot display human-readable titles (e.g. "Bose Headphones") and only reference internal mapped integer IDs.

## 2. Empty Configuration and Hardcoded Parameters
- **Incomplete**: `src/config/config.yaml` is empty.
- **Shortcoming**: 
  - Hyperparameters (`factors=64`, `regularization=0.01`, `iterations=25`, `alpha=40`), minimum thresholds (`min_reviews=10`), file paths, and maximum records (`50000`) are all hardcoded inside `main.py`.
  - There is no YAML parser to load configurations dynamically.

## 3. Empty Helpers Module
- **Incomplete**: `src/utils/helpers.py` is empty.
- **Shortcoming**: Common auxiliary tasks (such as logging, plotting training curves, mapping IDs to ASINs/Titles, and model serialization) have no helper functions defined.

## 4. Evaluation Inconsistencies & Limitations
- **Shortcoming**:
  - In `main.py`, ALS is evaluated using a subset of users (`max_users=2000`) for performance reasons, whereas the Popularity Baseline is evaluated over the entire dataset. This leads to inconsistent and statistically skewed comparisons.
  - Evaluation metrics are limited to basic Precision@K and Recall@K. Normalized Discounted Cumulative Gain (NDCG) and Mean Average Precision (MAP) are missing.

## 5. Scalability & Performance Issues
- **Shortcoming**:
  - The dataset files are extremely large (Reviews: 6.47 GB, Metadata: 1.31 GB). Reading them line-by-line using standard Python gzip + JSON parsing is slow and memory-intensive.
  - The current pipeline caps records at `50,000` to run quickly, meaning the model trains on a tiny fraction of the actual data.
  - Using a sparse representation `lil_matrix` during building and converting it to `csr_matrix` is correct, but iterating over the DataFrame rows with `.iterrows()` is highly inefficient for large datasets.

## 6. Lack of Model Persistence (Save/Load)
- **Incomplete**: There is no functionality to serialize/save the trained ALS model or the user/item ID mapping dictionaries (`user_map`, `item_map`) to disk.
- **Shortcoming**: Every time the system runs, it must reload the raw data, clean it, and retrain the model from scratch to make recommendations.

## 7. No Application Interface (API or CLI)
- **Incomplete**: The helper function `recommend_items` is defined in `main.py` but never called or exposed to the user.
- **Shortcoming**: There is no command-line interface or API (e.g., Flask/FastAPI) to query the recommender system for a given user ID.

## 8. Missing Documentation and Tests
- **Incomplete**: `README.md` is empty.
- **Shortcoming**:
  - No instructions on how to set up, configure, run, or verify the project.
  - Complete lack of unit, integration, or regression tests to verify data pipelines, preprocessing rules, or recommendation algorithms.
