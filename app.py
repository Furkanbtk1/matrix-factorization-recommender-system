"""
SVD-Based Matrix Factorization for Movie Recommendation
Capstone Project - Mathematics & Machine Learning
MovieLens 100K Dataset
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SVD Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #3ECFCF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #6C63FF;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #6C63FF;
        border-bottom: 2px solid #6C63FF22;
        padding-bottom: 4px;
        margin-top: 1.5rem;
    }
    .math-box {
        background: #13131f;
        border: 1px solid #6C63FF44;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.88rem;
        color: #cdd6f4;
    }
    .highlight {
        background: #6C63FF22;
        border-radius: 6px;
        padding: 2px 8px;
        color: #cba6f7;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(data_path="ml-100k/u.data", item_path="ml-100k/u.item"):
    col_names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(data_path, sep="\t", names=col_names)

    item_cols = ["item_id", "title", "release_date", "video_date", "url"] + [f"g{i}" for i in range(19)]
    items = pd.read_csv(item_path, sep="|", names=item_cols, encoding="latin-1")
    items = items[["item_id", "title"]]
    items["title"] = items["title"].str.strip()

    return df, items

@st.cache_data
def build_matrix(df):
    R = df.pivot_table(index="user_id", columns="item_id", values="rating")
    return R

@st.cache_data
def get_title_map(items):
    return dict(zip(items["item_id"], items["title"]))

# ─────────────────────────────────────────────
# SVD CORE
# ─────────────────────────────────────────────
class SVDRecommender:
    def __init__(self, n_factors=50, n_epochs=30, lr=0.005, reg=0.02, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.train_losses = []
        self.test_losses  = []

    def fit(self, R_train, R_test=None):
        np.random.seed(self.seed)
        self.R_train = R_train
        self.global_mean = np.nanmean(R_train)

        n_users, n_items = R_train.shape
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P  = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q  = np.random.normal(0, 0.1, (n_items, self.n_factors))

        known = [(u, i) for u in range(n_users)
                          for i in range(n_items)
                          if not np.isnan(R_train[u, i])]

        for epoch in range(self.n_epochs):
            np.random.shuffle(known)
            epoch_loss = 0.0
            for u, i in known:
                r_ui = R_train[u, i]
                pred = self._predict_raw(u, i)
                err  = r_ui - pred

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                puf = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * puf      - self.reg * self.Q[i])
                epoch_loss += err ** 2

            rmse_train = np.sqrt(epoch_loss / len(known))
            self.train_losses.append(rmse_train)

            if R_test is not None:
                test_known = [(u, i) for u in range(n_users)
                                      for i in range(n_items)
                                      if not np.isnan(R_test[u, i])]
                preds = [self._predict_raw(u, i) for u, i in test_known]
                actuals = [R_test[u, i] for u, i in test_known]
                rmse_test = np.sqrt(mean_squared_error(actuals, preds))
                self.test_losses.append(rmse_test)

        return self

    def _predict_raw(self, u, i):
        pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return float(np.clip(pred, 1, 5))

    def predict(self, user_idx, item_idx):
        return self._predict_raw(user_idx, item_idx)

    def get_full_matrix(self):
        return (self.global_mean
                + self.bu[:, None]
                + self.bi[None, :]
                + self.P @ self.Q.T)

    def recommend(self, user_idx, R_train, title_map, item_id_map, n=10):
        rated_items = set(np.where(~np.isnan(R_train[user_idx]))[0])
        scores = {}
        for i_idx in range(R_train.shape[1]):
            if i_idx not in rated_items:
                item_id = item_id_map[i_idx]
                scores[item_id] = self._predict_raw(user_idx, i_idx)
        top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        result = []
        for item_id, score in top_n:
            result.append({
                "Movie": title_map.get(item_id, f"Film {item_id}"),
                "Predicted Rating": round(score, 3)
            })
        return pd.DataFrame(result)


def truncated_svd_impute(R_full, k):
    """Classic truncated SVD for missing value imputation."""
    R_filled = R_full.copy()
    col_means = np.nanmean(R_filled, axis=0)
    inds = np.where(np.isnan(R_filled))
    R_filled[inds] = np.take(col_means, inds[1])

    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    U_k  = U[:, :k]
    s_k  = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    R_approx = U_k @ s_k @ Vt_k
    return np.clip(R_approx, 1, 5), s


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    st.markdown("---")

    n_factors = st.slider("Latent Factors (k)", 5, 100, 50, 5,
                          help="Number of latent dimensions in factorization. Higher = more expressive but slower.")
    n_epochs  = st.slider("Training Epochs", 5, 100, 30, 5,
                          help="SGD iterations over training data.")
    lr        = st.select_slider("Learning Rate", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.005)
    reg       = st.select_slider("Regularization (λ)", options=[0.005, 0.01, 0.02, 0.05, 0.1], value=0.02)
    test_size = st.slider("Test Split (%)", 10, 30, 20, 5)

    st.markdown("---")
    st.markdown("**📐 SVD Variant**")
    method = st.radio("", ["Matrix Factorization (SGD)", "Truncated SVD (Classic)"], index=0)

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Capstone Project · Mathematics Dept.\nMovieLens 100K Dataset")


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 SVD-Based Movie Recommender</div>', unsafe_allow_html=True)
st.markdown("**Matrix Factorization with Singular Value Decomposition** — Capstone Project")
st.markdown("---")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer",
    "🧮 Mathematical Background",
    "🏋️ Training & Evaluation",
    "🎯 Recommendations",
    "🔬 SVD Analysis"
])

# ─── TAB 1: Data Explorer ───────────────────
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    try:
        df, items = load_data()
        R = build_matrix(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👤 Users", f"{R.shape[0]:,}")
        c2.metric("🎬 Movies", f"{R.shape[1]:,}")
        c3.metric("⭐ Ratings", f"{len(df):,}")
        c4.metric("❓ Sparsity", f"{(R.isna().sum().sum() / R.size * 100):.1f}%")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Rating Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#13131f")
            ax.set_facecolor("#13131f")
            counts = df["rating"].value_counts().sort_index()
            bars = ax.bar(counts.index, counts.values,
                          color=["#6C63FF","#7c75ff","#5a53dd","#9b94ff","#4a45bb"])
            ax.set_xlabel("Rating", color="#cdd6f4")
            ax.set_ylabel("Count", color="#cdd6f4")
            ax.tick_params(colors="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                        f"{val:,}", ha="center", fontsize=8, color="#cdd6f4")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("**Ratings per User (Top 50)**")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#13131f")
            ax.set_facecolor("#13131f")
            user_counts = df.groupby("user_id").size().sort_values(ascending=False).head(50)
            ax.plot(range(len(user_counts)), user_counts.values, color="#3ECFCF", linewidth=1.5)
            ax.fill_between(range(len(user_counts)), user_counts.values, alpha=0.3, color="#3ECFCF")
            ax.set_xlabel("User Rank", color="#cdd6f4")
            ax.set_ylabel("# Ratings", color="#cdd6f4")
            ax.tick_params(colors="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Sample Rating Matrix (first 15 users × 20 movies)**")
        sample = R.iloc[:15, :20].copy()
        # Film ID'leri yerine kısa film isimlerini kullan
        item_id_to_title = dict(zip(items["item_id"], items["title"]))
        short_titles = [item_id_to_title.get(c, str(c))[:18] for c in sample.columns]
        sample.columns = short_titles

        fig, ax = plt.subplots(figsize=(14, 5), facecolor="#13131f")
        ax.set_facecolor("#13131f")
        mask = sample.isna()
        im = ax.imshow(sample.fillna(0), cmap="Blues", aspect="auto", vmin=0, vmax=5)
        for (row, col), val in np.ndenumerate(mask.values):
            if val:
                ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                             color="#1a1a2e", zorder=2))
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(short_titles)))
        ax.set_xticklabels(short_titles, rotation=45, ha="right", fontsize=7, color="#cdd6f4")
        ax.set_ylabel("User ID", color="#cdd6f4")
        ax.tick_params(axis="y", colors="#cdd6f4")
        ax.set_title("⬛ = Missing (NaN)  |  Renkli = Verilen Rating", color="#cdd6f4", pad=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Top 10 Most Rated Movies**")
        top_movies = (df.groupby("item_id")
                        .agg(count=("rating","count"), avg=("rating","mean"))
                        .reset_index()
                        .merge(items, on="item_id")
                        .sort_values("count", ascending=False)
                        .head(10)[["title","count","avg"]]
                        .rename(columns={"count":"# Ratings","avg":"Avg Rating"}))
        top_movies["Avg Rating"] = top_movies["Avg Rating"].round(2)
        st.dataframe(top_movies, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("📂 Dataset not found. Please make sure `ml-100k/` folder is in the same directory as `app.py`.")


# ─── TAB 2: Math Background ─────────────────
with tab2:
    st.markdown('<div class="section-header">Mathematical Foundations</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. The Problem: Incomplete Rating Matrix

    We have a **user–item matrix** R ∈ ℝ^(m×n) where m = number of users, n = number of items.
    Most entries are **missing** (NaN) — only a small fraction of movies have been rated.

    **Goal:** Predict the missing entries R̂ᵤᵢ ≈ rᵤᵢ.
    """)

    st.markdown('<div class="math-box">R ≈ R̂ = μ + bᵤ + bᵢ + PQ^T</div>', unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 2. Singular Value Decomposition (SVD)

    For a matrix A ∈ ℝ^(m×n):
    """)

    st.markdown('<div class="math-box">A = U · Σ · Vᵀ</div>', unsafe_allow_html=True)
    st.markdown("""
    - **U** ∈ ℝ^(m×m) — left singular vectors (user latent space)
    - **Σ** ∈ ℝ^(m×n) — diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
    - **Vᵀ** ∈ ℝ^(n×n) — right singular vectors (item latent space)

    ---
    ### 3. Truncated SVD (Rank-k Approximation)

    The **Eckart–Young–Mirsky theorem** states that the best rank-k approximation is:
    """)

    st.markdown('<div class="math-box">A_k = U_k · Σ_k · V_kᵀ = Σ (i=1..k) σᵢ · uᵢ · vᵢᵀ</div>', unsafe_allow_html=True)
    st.markdown("""
    This minimizes the Frobenius norm: ‖A − A_k‖_F is minimized over all rank-k matrices.

    ---
    ### 4. Matrix Factorization Model

    Instead of full SVD, we learn two low-rank factor matrices:
    """)

    st.markdown('<div class="math-box">R̂ᵤᵢ = μ + bᵤ + bᵢ + Pᵤ · Qᵢᵀ</div>', unsafe_allow_html=True)
    st.markdown("""
    where:
    - μ = global mean rating
    - bᵤ = user bias (does this user rate higher/lower than average?)
    - bᵢ = item bias (is this movie rated higher/lower than average?)
    - **P** ∈ ℝ^(m×k) — user factor matrix
    - **Q** ∈ ℝ^(n×k) — item factor matrix

    ---
    ### 5. Loss Function

    We minimize regularized squared error over known ratings:
    """)

    st.markdown("""
    <div class="math-box">
    L = Σ_{(u,i)∈K} (rᵤᵢ − R̂ᵤᵢ)² + λ(‖Pᵤ‖² + ‖Qᵢ‖² + bᵤ² + bᵢ²)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **K** = set of known (user, item) pairs
    - **λ** = regularization coefficient (prevents overfitting)

    ---
    ### 6. SGD Update Rules

    Using Stochastic Gradient Descent with learning rate η:
    """)

    st.markdown("""
    <div class="math-box">
    eᵤᵢ = rᵤᵢ − R̂ᵤᵢ  (prediction error)
    <br><br>
    bᵤ ← bᵤ + η · (eᵤᵢ − λ·bᵤ)
    <br>
    bᵢ ← bᵢ + η · (eᵤᵢ − λ·bᵢ)
    <br>
    Pᵤ ← Pᵤ + η · (eᵤᵢ·Qᵢ − λ·Pᵤ)
    <br>
    Qᵢ ← Qᵢ + η · (eᵤᵢ·Pᵤ − λ·Qᵢ)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 7. Evaluation Metrics

    | Metric | Formula | Interpretation |
    |--------|---------|---------------|
    | RMSE | √(Σ(r − r̂)²/n) | Lower is better; in rating units |
    | MAE  | Σ\|r − r̂\|/n  | Mean absolute error per rating |
    | Coverage | \|predicted\| / \|total\| | Fraction of items recommendable |
    """)


# ─── TAB 3: Training ────────────────────────
with tab3:
    st.markdown('<div class="section-header">Model Training & Evaluation</div>', unsafe_allow_html=True)

    if train_btn:
        try:
            df, items = load_data()
            R_full = build_matrix(df)

            # Train/test split at rating level
            with st.spinner("Splitting data & preparing matrices…"):
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(df, test_size=test_size/100, random_state=42)

                R_train_df = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
                # Align columns/index
                R_train_mat = R_train_df.reindex(index=R_full.index, columns=R_full.columns).values.astype(float)

                R_test_df = test_df.pivot_table(index="user_id", columns="item_id", values="rating")
                R_test_mat = R_test_df.reindex(index=R_full.index, columns=R_full.columns).values.astype(float)

                # Set train entries to NaN in test and vice versa
                for idx, row in test_df.iterrows():
                    u_idx = list(R_full.index).index(row["user_id"])
                    i_idx = list(R_full.columns).index(row["item_id"])
                    R_train_mat[u_idx, i_idx] = np.nan  # remove from train

                for idx, row in train_df.iterrows():
                    u_idx = list(R_full.index).index(row["user_id"])
                    i_idx = list(R_full.columns).index(row["item_id"])
                    R_test_mat[u_idx, i_idx] = np.nan  # remove from test

            if method == "Matrix Factorization (SGD)":
                progress = st.progress(0, text="Training SVD Matrix Factorization…")
                model = SVDRecommender(n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg=reg)

                # Patch fit to update progress
                np.random.seed(42)
                model.global_mean = np.nanmean(R_train_mat)
                n_users, n_items = R_train_mat.shape
                model.bu = np.zeros(n_users)
                model.bi = np.zeros(n_items)
                model.P  = np.random.normal(0, 0.1, (n_users, n_factors))
                model.Q  = np.random.normal(0, 0.1, (n_items, n_factors))
                model.train_losses = []
                model.test_losses  = []
                model.R_train = R_train_mat

                known = [(u, i) for u in range(n_users)
                                  for i in range(n_items)
                                  if not np.isnan(R_train_mat[u, i])]
                test_known = [(u, i) for u in range(n_users)
                                      for i in range(n_items)
                                      if not np.isnan(R_test_mat[u, i])]

                for epoch in range(n_epochs):
                    np.random.shuffle(known)
                    epoch_loss = 0.0
                    for u, i in known:
                        r_ui = R_train_mat[u, i]
                        pred = float(np.clip(model.global_mean + model.bu[u] + model.bi[i] + model.P[u] @ model.Q[i], 1, 5))
                        err  = r_ui - pred
                        model.bu[u] += lr * (err - reg * model.bu[u])
                        model.bi[i] += lr * (err - reg * model.bi[i])
                        puf = model.P[u].copy()
                        model.P[u] += lr * (err * model.Q[i] - reg * model.P[u])
                        model.Q[i] += lr * (err * puf       - reg * model.Q[i])
                        epoch_loss += err ** 2
                    model.train_losses.append(np.sqrt(epoch_loss / len(known)))

                    preds = [float(np.clip(model.global_mean + model.bu[u] + model.bi[i] + model.P[u] @ model.Q[i], 1, 5))
                             for u, i in test_known]
                    actuals = [R_test_mat[u, i] for u, i in test_known]
                    model.test_losses.append(np.sqrt(mean_squared_error(actuals, preds)))
                    progress.progress((epoch + 1) / n_epochs, text=f"Epoch {epoch+1}/{n_epochs} — Train RMSE: {model.train_losses[-1]:.4f}")

                progress.empty()
                st.session_state["model"]       = model
                st.session_state["R_full"]      = R_full
                st.session_state["R_train_mat"] = R_train_mat
                st.session_state["R_test_mat"]  = R_test_mat
                st.session_state["items"]       = items
                st.session_state["df"]          = df
                st.session_state["user_ids"]    = list(R_full.index)
                st.session_state["item_ids"]    = list(R_full.columns)

                final_train = model.train_losses[-1]
                final_test  = model.test_losses[-1]
                mae = np.mean(np.abs(np.array(preds) - np.array(actuals)))

                st.success(f"✅ Training complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Train RMSE", f"{final_train:.4f}")
                c2.metric("Test RMSE",  f"{final_test:.4f}")
                c3.metric("Test MAE",   f"{mae:.4f}")

                # Learning curve
                fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#13131f")
                ax.set_facecolor("#13131f")
                ax.plot(model.train_losses, label="Train RMSE", color="#6C63FF", linewidth=2)
                ax.plot(model.test_losses,  label="Test RMSE",  color="#3ECFCF", linewidth=2)
                ax.set_xlabel("Epoch", color="#cdd6f4")
                ax.set_ylabel("RMSE", color="#cdd6f4")
                ax.set_title("Learning Curve", color="#cdd6f4")
                ax.legend(facecolor="#1E1E2E", labelcolor="#cdd6f4")
                ax.tick_params(colors="#cdd6f4")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333355")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Error distribution
                errors = np.array(preds) - np.array(actuals)
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#13131f")
                for ax in axes:
                    ax.set_facecolor("#13131f")
                    ax.tick_params(colors="#cdd6f4")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#333355")

                axes[0].hist(errors, bins=50, color="#6C63FF", edgecolor="#13131f", alpha=0.8)
                axes[0].axvline(0, color="#3ECFCF", linewidth=2, linestyle="--")
                axes[0].set_xlabel("Prediction Error", color="#cdd6f4")
                axes[0].set_title("Error Distribution", color="#cdd6f4")

                axes[1].scatter(actuals[:2000], preds[:2000], alpha=0.15, s=8, color="#6C63FF")
                axes[1].plot([1,5],[1,5], color="#3ECFCF", linewidth=2, linestyle="--")
                axes[1].set_xlabel("Actual Rating", color="#cdd6f4")
                axes[1].set_ylabel("Predicted Rating", color="#cdd6f4")
                axes[1].set_title("Predicted vs Actual", color="#cdd6f4")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            else:  # Truncated SVD
                with st.spinner("Running Truncated SVD…"):
                    R_filled = R_full.values.copy().astype(float)
                    R_approx, singular_vals = truncated_svd_impute(R_filled, n_factors)

                test_known = [(u, i) for u in range(R_test_mat.shape[0])
                                      for i in range(R_test_mat.shape[1])
                                      if not np.isnan(R_test_mat[u, i])]
                preds   = [R_approx[u, i] for u, i in test_known]
                actuals = [R_test_mat[u, i] for u, i in test_known]
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                mae  = np.mean(np.abs(np.array(preds) - np.array(actuals)))

                st.success("✅ SVD decomposition complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Test RMSE", f"{rmse:.4f}")
                c2.metric("Test MAE",  f"{mae:.4f}")
                c3.metric("Rank-k used", f"{n_factors}")

                st.session_state["svd_approx"]  = R_approx
                st.session_state["svd_svals"]   = singular_vals
                st.session_state["R_full"]      = R_full
                st.session_state["items"]       = items
                st.session_state["df"]          = df
                st.session_state["user_ids"]    = list(R_full.index)
                st.session_state["item_ids"]    = list(R_full.columns)

                # Explained variance
                fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#13131f")
                ax.set_facecolor("#13131f")
                cumvar = np.cumsum(singular_vals[:50]**2) / np.sum(singular_vals**2) * 100
                ax.plot(range(1, 51), cumvar, color="#6C63FF", linewidth=2, marker="o", markersize=3)
                ax.axhline(90, color="#3ECFCF", linestyle="--", linewidth=1, label="90% threshold")
                ax.set_xlabel("Number of Singular Values", color="#cdd6f4")
                ax.set_ylabel("Cumulative Variance (%)", color="#cdd6f4")
                ax.set_title("Explained Variance by Singular Values", color="#cdd6f4")
                ax.tick_params(colors="#cdd6f4")
                ax.legend(facecolor="#1E1E2E", labelcolor="#cdd6f4")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333355")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        except FileNotFoundError:
            st.error("Dataset not found. Please ensure `ml-100k/` folder is present.")
    else:
        st.info("👈 Configure parameters in the sidebar and click **🚀 Train Model** to begin.")


# ─── TAB 4: Recommendations ─────────────────
with tab4:
    st.markdown('<div class="section-header">Personalized Recommendations</div>', unsafe_allow_html=True)

    if "model" in st.session_state:
        model   = st.session_state["model"]
        R_full  = st.session_state["R_full"]
        R_train = st.session_state["R_train_mat"]
        items   = st.session_state["items"]
        user_ids = st.session_state["user_ids"]
        item_ids = st.session_state["item_ids"]

        title_map = get_title_map(items)

        col1, col2 = st.columns([1, 2])
        with col1:
            selected_user = st.selectbox("👤 Kullanıcı Seç (User ID)", user_ids[:100])
            n_recs = st.slider("🎬 Öneri Sayısı", 5, 20, 10)
            rec_btn = st.button("🔍 Önerileri Getir", type="primary", use_container_width=True)

            # Kullanıcı istatistikleri
            u_idx = user_ids.index(selected_user)
            rated_count = int(np.sum(~np.isnan(R_train[u_idx])))
            avg_rating  = float(np.nanmean(R_train[u_idx]))
            st.markdown("---")
            st.metric("Toplam Rating", f"{rated_count} film")
            st.metric("Ortalama Rating", f"{avg_rating:.2f} ⭐")

        if rec_btn:
            recs = model.recommend(u_idx, R_train, title_map, item_ids, n=n_recs)

            with col2:
                st.markdown(f"**🎯 User {selected_user} için Top {n_recs} Öneri**")
                fig, ax = plt.subplots(figsize=(8, n_recs * 0.42 + 0.8), facecolor="#13131f")
                ax.set_facecolor("#13131f")
                bar_colors = plt.cm.plasma(np.linspace(0.3, 0.85, len(recs)))
                # Film isimlerini kısalt
                short_movies = [m[:35] + "…" if len(m) > 35 else m for m in recs["Movie"]]
                bars = ax.barh(short_movies, recs["Predicted Rating"], color=bar_colors, height=0.6)
                # Rating değerlerini bar üzerine yaz
                for bar, val in zip(bars, recs["Predicted Rating"]):
                    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height()/2,
                            f"{val:.2f}", va="center", fontsize=8, color="#cdd6f4")
                ax.set_xlim(1, 5.8)
                ax.axvline(4, color="#3ECFCF", linestyle="--", alpha=0.5, linewidth=1, label="Rating = 4")
                ax.set_xlabel("Tahmin Edilen Rating", color="#cdd6f4")
                ax.tick_params(colors="#cdd6f4", labelsize=8)
                ax.legend(facecolor="#1E1E2E", labelcolor="#cdd6f4", fontsize=8)
                ax.invert_yaxis()
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333355")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Kullanıcının daha önce izlediği filmler — tam liste
            rated_mask  = ~np.isnan(R_train[u_idx])
            rated_items = np.where(rated_mask)[0]
            rated_df = pd.DataFrame({
                "🎬 Film Adı": [title_map.get(item_ids[i], f"Film {item_ids[i]}")
                                for i in rated_items],
                "⭐ Verilen Rating": [int(R_train[u_idx, i]) for i in rated_items]
            }).sort_values("⭐ Verilen Rating", ascending=False).reset_index(drop=True)

            st.markdown(f"**📋 User {selected_user}'nin İzlediği Tüm Filmler ({len(rated_df)} adet)**")
            st.dataframe(rated_df, use_container_width=True, hide_index=True, height=300)

    elif "svd_approx" in st.session_state:
        R_approx = st.session_state["svd_approx"]
        R_full   = st.session_state["R_full"]
        items    = st.session_state["items"]
        user_ids = st.session_state["user_ids"]
        item_ids = st.session_state["item_ids"]

        selected_user = st.selectbox("Select User ID", user_ids[:100])
        n_recs = st.slider("Number of Recommendations", 5, 20, 10)

        u_idx = user_ids.index(selected_user)
        rated_mask = ~np.isnan(R_full.values[u_idx])
        unrated = np.where(~rated_mask)[0]

        scores = [(item_ids[i], R_approx[u_idx, i]) for i in unrated]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:n_recs]
        recs = pd.DataFrame([{
            "Movie": items[items.item_id == iid]["title"].values[0] if len(items[items.item_id == iid]) > 0 else f"Item {iid}",
            "Predicted Rating": round(s, 3)
        } for iid, s in scores])
        st.dataframe(recs, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Train the model first in the **Training & Evaluation** tab.")


# ─── TAB 5: SVD Analysis ────────────────────
with tab5:
    st.markdown('<div class="section-header">SVD Deep Dive</div>', unsafe_allow_html=True)

    if "model" in st.session_state or "svd_approx" in st.session_state:
        df, items = load_data()
        R_full = build_matrix(df)
        R_filled = R_full.fillna(R_full.mean()).values.astype(float)
        U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Singular Value Spectrum**")
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#13131f")
            ax.set_facecolor("#13131f")
            ax.bar(range(1, 31), s[:30], color="#6C63FF")
            ax.set_xlabel("Index", color="#cdd6f4")
            ax.set_ylabel("Singular Value", color="#cdd6f4")
            ax.tick_params(colors="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Cumulative Explained Variance**")
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#13131f")
            ax.set_facecolor("#13131f")
            cum_var = np.cumsum(s**2) / np.sum(s**2) * 100
            ax.plot(range(1, 101), cum_var[:100], color="#3ECFCF", linewidth=2)
            ax.fill_between(range(1, 101), cum_var[:100], alpha=0.2, color="#3ECFCF")
            for threshold, color in [(50, "#f38ba8"), (75, "#fab387"), (90, "#a6e3a1")]:
                k = np.argmax(cum_var >= threshold) + 1
                ax.axhline(threshold, color=color, linestyle="--", alpha=0.7, linewidth=1,
                           label=f"{threshold}% @ k={k}")
            ax.set_xlabel("k (number of components)", color="#cdd6f4")
            ax.set_ylabel("Cumulative Variance (%)", color="#cdd6f4")
            ax.legend(facecolor="#1E1E2E", labelcolor="#cdd6f4", fontsize=8)
            ax.tick_params(colors="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**User Latent Space (First 2 Principal Components)**")
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#13131f")
        ax.set_facecolor("#13131f")
        ax.scatter(U[:, 0], U[:, 1], alpha=0.35, s=12, c=U[:, 2], cmap="plasma")
        ax.set_xlabel("1st Left Singular Vector (u₁)", color="#cdd6f4")
        ax.set_ylabel("2nd Left Singular Vector (u₂)", color="#cdd6f4")
        ax.set_title("Users in Latent Factor Space", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Matrix reconstruction comparison
        st.markdown("**Matrix Reconstruction Quality vs. Rank k**")
        sample_R = R_filled[:50, :50]
        U2, s2, Vt2 = np.linalg.svd(sample_R, full_matrices=False)
        ks = [1, 5, 10, 20, 50]
        errors = []
        for k in ks:
            R_k = U2[:, :k] @ np.diag(s2[:k]) @ Vt2[:k, :]
            errors.append(np.linalg.norm(sample_R - R_k, "fro"))

        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#13131f")
        ax.set_facecolor("#13131f")
        ax.plot(ks, errors, color="#6C63FF", linewidth=2, marker="o", markersize=8)
        for k, e in zip(ks, errors):
            ax.annotate(f"k={k}\n{e:.1f}", (k, e), textcoords="offset points",
                        xytext=(0, 10), ha="center", color="#cdd6f4", fontsize=8)
        ax.set_xlabel("Rank k", color="#cdd6f4")
        ax.set_ylabel("Frobenius Error ‖R − Rₖ‖_F", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👈 Train the model first to unlock SVD Analysis.")

# Footer
st.markdown("---")
st.caption("Capstone Project · SVD-Based Matrix Factorization · MovieLens 100K · Mathematics Department")