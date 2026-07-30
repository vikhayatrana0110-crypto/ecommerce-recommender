# Project Context: E-commerce Recommender System

This project is a collaborative filtering recommender system designed for e-commerce, specifically using the Amazon Electronics reviews dataset. It leverages implicit feedback modeling with Alternating Least Squares (ALS) and compares it against a popularity-based baseline.

## Project Structure

- **`main.py`**: The entry point of the project. It orchestrates the loading, preprocessing, split, training, evaluation, and logging of statistics.
- **`requirements.txt`**: Project dependencies, including `pandas`, `numpy`, `scikit-learn`, `implicit`, `scipy`, `tqdm`, `pyyaml`, `matplotlib`, and `pyarrow`.
- **`src/`**: Source folder.
  - **`data/`**:
    - [load_data.py](file:///d:/%231%20PROJECT/ecommerce-recommender/src/data/load_data.py): Handles streaming, reading, and loading raw review and metadata `.jsonl.gz` datasets into Pandas DataFrames.
    - [preprocess.py](file:///d:/%231%20PROJECT/ecommerce-recommender/src/data/preprocess.py): Functions for cleaning reviews, filtering inactive users, filtering unpopular items, and building sparse user-item interaction matrices.
  - **`config/`**:
    - [config.yaml](file:///d:/%231%20PROJECT/ecommerce-recommender/src/config/config.yaml): Directory for configuration parameters (currently empty).
  - **`utils/`**:
    - [helpers.py](file:///d:/%231%20PROJECT/ecommerce-recommender/src/utils/helpers.py): Helper utilities (currently empty).
- **`data/raw/`**: Contains the raw datasets (not tracked by Git due to size):
  - `Electronics.jsonl.gz` (Reviews dataset)
  - `meta_Electronics.jsonl.gz` (Metadata dataset)

## Core Technologies
1. **Python 3.x**
2. **Implicit**: Used to train the Alternating Least Squares (ALS) model.
3. **SciPy**: For creating and handling sparse matrices (CSR/LIL) representing user-item interactions.
4. **Pandas/NumPy**: For data loading, manipulation, and computation.
