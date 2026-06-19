# 🎬 SVD-Based Matrix Factorization for Collaborative Filtering

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A complete implementation of **Singular Value Decomposition (SVD)** and **gradient-based Matrix Factorization** for missing-value imputation in sparse rating matrices — built as a Mathematics Department Capstone Project.

---

## 📖 Overview

This project tackles the **collaborative filtering problem**: given a sparse user-item rating matrix where over 93% of entries are missing, predict the unobserved ratings to generate personalized recommendations.

Two complementary approaches are implemented and compared:

| Method | Description |
|---|---|
| **Funk SVD (Matrix Factorization)** | Learns latent factor matrices `P` and `Q` directly from observed ratings via Stochastic Gradient Descent — no imputation required |
| **Truncated SVD (Classical)** | Applies the Eckart-Young-Mirsky optimal low-rank approximation after mean-imputing missing entries |

The project includes a full **interactive web application** and a **mathematically rigorous report** with properly validated experimental results (train/validation/test methodology).

---

## 🧮 Mathematical Foundation

The rating prediction model:

```
R̂(u,i) = μ + bᵤ + bᵢ + Pᵤ · Qᵢᵀ
```

where:
- `μ` — global mean rating
- `bᵤ`, `bᵢ` — user and item bias terms
- `Pᵤ ∈ ℝᵏ` — latent factor vector for user u
- `Qᵢ ∈ ℝᵏ` — latent factor vector for item i

Optimized via regularized SGD minimizing:

```
L = Σ₍ᵤ,ᵢ₎∈K (R(u,i) − R̂(u,i))² + λ(‖Pᵤ‖² + ‖Qᵢ‖² + bᵤ² + bᵢ²)
```

Full derivation — including the **Eckart-Young-Mirsky theorem**, SVD decomposition `A = UΣVᵀ`, and complete SGD update rules — is provided in [`capstone_report_final.pdf`](capstone_report_final.pdf).

---

## 📊 Results

Evaluated on **MovieLens 100K** (943 users · 1,682 movies · 100,000 ratings) using a rigorous **70% train / 10% validation / 20% test** split — hyperparameters selected on validation set only, test set used exactly once.

| Model | Test RMSE | Test MAE | Improvement |
|---|---|---|---|
| Global Mean Baseline | 1.1260 | 0.9340 | — |
| User Mean Baseline | 1.0352 | 0.8220 | −8.1% |
| Truncated SVD (k=50) | 0.9740 | 0.7750 | −13.5% |
| **Matrix Factorization (k=50, 30 epochs)** | **0.9401** | **0.7365** | **−16.6%** |

### Optimal Hyperparameters (selected via validation RMSE)

| Parameter | Value |
|---|---|
| Latent factors (k) | 50 |
| Learning rate (η) | 0.005 |
| Regularization (λ) | 0.02 |
| Training epochs | 30 |

Full hyperparameter sweep (k, η, λ, epochs) with validation curves available in the report.

---

## 🖥️ Application Features

Built with **Streamlit**, the app includes 5 interactive tabs:

| Tab | Description |
|---|---|
| 📊 **Data Explorer** | Dataset statistics, rating distribution, sparsity heatmap with real movie titles |
| 🧮 **Mathematical Background** | Full SVD/MF theory rendered in-app |
| 🏋️ **Training & Evaluation** | Live model training with real-time RMSE/MAE tracking and learning curves |
| 🎯 **Recommendations** | Generate personalized top-N movie recommendations for any user |
| 🔬 **SVD Analysis** | Singular value spectrum, explained variance, latent space visualization |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/Furkanbtk1/matrix-factorization-recommender-system.git
cd matrix-factorization-recommender-system
pip install streamlit numpy pandas matplotlib seaborn scikit-learn reportlab
```

### Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### Regenerate the PDF report

```bash
python generate_report.py
```

---

## 📁 Project Structure

```
matrix-factorization-recommender-system/
├── app.py                       # Streamlit web application
├── generate_report.py           # PDF report generator
├── capstone_report_final.pdf    # Full academic report
├── ml-100k/                     # MovieLens 100K dataset
│   ├── u.data                   # Ratings (user_id, item_id, rating, timestamp)
│   ├── u.item                   # Movie metadata (titles, genres)
│   └── u.user                   # User demographics
└── README.md
```

---

## 🔬 Methodology Highlights

- **No data leakage**: imputation statistics and SVD decomposition computed strictly on the training matrix
- **Proper validation**: all hyperparameter selection performed on a held-out validation set, never on the test set
- **Reproducible**: fixed random seeds, fully specified preprocessing pipeline
- **Parameter accounting**: total model parameters formally derived as `(m+n)·k + m + n + 1`

---

## 📚 References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer, 42(8), 30–37.
2. Funk, S. (2006). *Netflix Update: Try This at Home*.
3. Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank*. Psychometrika.
4. Harper, F.M., & Konstan, J.A. (2015). *The MovieLens Datasets: History and Context*. ACM TIIS.
5. Golub, G.H., & Van Loan, C.F. (2013). *Matrix Computations* (4th ed.).

---

## 🎓 About

This project was developed as a **Capstone Project for the Department of Mathematics**, demonstrating the application of linear algebra (SVD), convex optimization (SGD), and statistical learning to a real-world recommendation system problem.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.