# E-Commerce Recommendation System  
### Implicit Collaborative Filtering using ALS

## 📌 Problem Statement

Build a scalable recommendation system that suggests top-N products to users based on implicit interaction data (views/purchases), and benchmark it against a popularity baseline.

The objective is to evaluate whether matrix factorization (ALS) improves ranking performance under sparse real-world conditions.

---

##  Key Highlights

- Built end-to-end collaborative filtering pipeline
- Implemented **Implicit ALS** using sparse matrix factorization
- Designed baseline model for performance comparison
- Evaluated using ranking metrics (Precision@10, Recall@10)
- Handled high sparsity interaction matrix (~1.8% density)
- Structured modular ML project with clean separation of concerns

---

## 📊 Dataset & Preprocessing

- Initial reviews: 50,000
- After filtering active users & popular items
- Final interaction matrix:

| Metric | Value |
|--------|-------|
| Users | 668 |
| Items | 84 |
| Matrix Shape | (668, 84) |
| Non-zero interactions | 1,384 |
| Density | 0.0185 |
| Avg interactions/user | 1.55 |

### Processing Steps
- Removed inactive users
- Removed low-frequency items
- Built sparse CSR interaction matrix
- Train/Test split on implicit feedback

---

##  Models Implemented

### 1️⃣ Popularity Baseline
- Recommends globally most interacted items
- Serves as benchmark for collaborative filtering

### 2️⃣ Implicit ALS (Alternating Least Squares)
- Library: `implicit`
- Confidence-weighted matrix factorization
- 64 latent factors
- Trained on sparse matrix using optimized linear algebra backend

Learned Representations:
- User factors: (668 × 64)
- Item factors: (84 × 64)

---

## 📈 Evaluation (Top-10 Ranking)

| Model | Precision@10 | Recall@10 |
|-------|--------------|-----------|
| ALS | **0.0206** | 0.1657 |
| Popularity | 0.0250 | **0.2500** |

---

## 🔎 Analysis & Insights

- Dataset is extremely sparse (~1.8% density).
- Majority of users have only 1 interaction.
- Popularity baseline outperformed ALS due to:
  - Limited personalization signal
  - Insufficient user interaction history
- Demonstrates importance of data density in collaborative filtering systems.

This mirrors real-world recommender challenges where model complexity does not guarantee better ranking performance under sparse conditions.

---

## 🏗 Project Architecture
ecommerce-recommender/
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── als_model.py
│ ├── evaluation.py
│
├── main.py
├── requirements.txt
└── README.md
