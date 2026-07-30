# E-Commerce Recommendation System  
### Implicit Collaborative Filtering using ALS with Streamlit Dashboard

This is an end-to-end collaborative filtering recommender system designed for the Amazon Electronics reviews dataset. It features dynamic configuration, optimized data ingestion, vectorized matrix operations, implicit ALS training, model persistence, evaluation metrics, unit testing, and a Streamlit UI dashboard.

---

## Problem Statement

Build a scalable recommendation system that suggests top-N products to users based on implicit interaction data (views/purchases), and benchmark it against a popularity baseline.

The objective is to evaluate whether matrix factorization (ALS) improves ranking performance under sparse real-world conditions.

---

## Features & Highlights

- **Built End-to-End Collaborative Filtering Pipeline**: Clean separation of data loading, preprocessing, model training, and evaluation.
- **Fast Chunked Loading**: Memory-efficient streaming of multi-gigabyte `.gz` compressed datasets using pandas chunking.
- **Vectorized Preprocessing**: Speeds up CSR interaction matrix generation by mapping indices in vector space (avoiding slow python loops).
- **Implicit ALS Matrix Factorization**: Latent factor model using the `implicit` library.
- **Pickled Model Persistence**: Serializes trained Alternating Least Squares (ALS) models and dictionary mappings for instant serving.
- **Streamlit Interactive UI**: Search users, view purchase histories, display recommendations, and handle new/unknown user IDs via a built-in popularity-based **Cold Start** fallback mechanism.
- **Comprehensive Unit Testing**: Automated tests via `pytest` for pipeline transformations and train-test splits.

---

## Dataset & Preprocessing

- Initial reviews: 50,000
- After filtering active users & popular items
- Final interaction matrix:

| Metric | Value |
|--------|-------|
| Users | 668 |
| Items | 84 |
| Matrix Shape | (668, 84) |
| Non-zero interactions | 1,384 |
| Density | 0.0185 (1.85%) |
| Avg interactions/user | 1.55 |

### Processing Steps
- Remove inactive users (e.g., `< 10` reviews)
- Remove low-frequency items (e.g., `< 10` reviews)
- Build sparse CSR interaction matrix
- Train/Test split on implicit feedback

---

## Project Architecture

```text
ecommerce-recommender/
│
├── src/
│   ├── config/
│   │   └── config.yaml           # Hyperparameters and file path configurations
│   ├── data/
│   │   ├── load_data.py          # Data ingestion from compressed source files
│   │   └── preprocess.py         # Vectorized matrix preparation and filtering
│   └── utils/
│       └── helpers.py            # Utility functions for YAML config and pickle serialization
│
├── tests/
│   └── test_recommender.py       # Unit tests for verification
│
├── data/                         # Locally stored data files (Ignored from Git)
│   ├── raw/
│   │   ├── Electronics.jsonl.gz
│   │   └── meta_Electronics.jsonl.gz
│   ├── model.pkl
│   ├── user_map.pkl
│   └── item_map.pkl
│
├── app.py                        # Streamlit Interactive Dashboard UI
├── main.py                       # Orchestrator script for training and evaluation
├── requirements.txt              # Project package dependencies
└── README.md                     # Documentation
```

---

## Installation & Setup

1. **Activate the Virtual Environment**:
   ```bash
   .\venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### 1. Run Model Training & Evaluation
To parse raw datasets, pre-process matrices, fit the ALS model, evaluate baseline metrics, and serialize artifacts to the `data/` folder, run:
```bash
python main.py
```

### 2. Launch the Streamlit Dashboard
To launch the interactive web interface, run:
```bash
streamlit run app.py
```

### 3. Run Unit Tests
To run verification tests:
```bash
python -m pytest
```

---

## Configuration Settings
All paths and hyperparameters can be tweaked dynamically in [config.yaml](file:///d:/%231%20PROJECT/ecommerce-recommender/src/config/config.yaml):
- **Filtering thresholds** (e.g., minimum user/item reviews)
- **Model parameters** (factors, regularization, iterations, confidence alpha)
- **Record caps** (e.g., number of review records to read)
- **File paths** (source datasets and serialized model files)

---

## Evaluation (Top-10 Ranking)

| Model | Precision@10 | Recall@10 |
|-------|--------------|-----------|
| **ALS** | **0.0206** | 0.1657 |
| Popularity | 0.0250 | **0.2500** |

---

## Analysis & Insights

- Dataset is extremely sparse (~1.8% density).
- Majority of users have only 1 interaction.
- Popularity baseline outperformed ALS due to:
  - Limited personalization signal
  - Insufficient user interaction history
- Demonstrates the critical importance of data density in collaborative filtering systems.

This mirrors real-world recommender challenges where model complexity does not guarantee better ranking performance under sparse conditions.
