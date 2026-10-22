"""
SHAP Value Analysis on the V3 Stage-2 LightGBM Regressor (Mục 4 - Insights, Rubric 1.5đ)

Mục tiêu: giải thích mô hình hồi quy (Stage 2 Regressor) của pipeline V3 bằng
SHAP (SHapley Additive exPlanations) — trả lời câu hỏi "đặc trưng nào đóng góp
nhiều nhất vào điểm priority dự đoán, và theo chiều hướng nào?".

Pipeline:
  1. Load train.csv + SBERT embeddings đã cache (train_embeddings_minilm.npy).
  2. Tái tạo đúng feature matrix X_full (67 cột: 66 cột V3 gốc +
     author_global_preferred_score) bằng extract_features_v3(is_train=True),
     giống hệt bước "Training on Full Dataset" trong train_v3.py.
  3. Train một LGBMRegressor Stage 2 trên toàn bộ train.csv (cùng hyperparameter
     với build_stage2_models() trong train_v3.py).
  4. Tính SHAP values bằng shap.TreeExplainer trên 500 mẫu đầu (để nhanh).
  5. Xuất 2 biểu đồ vào reports/figures/:
       - 06_shap_summary.png: beeswarm summary plot, top 20 features.
       - 07_shap_bar.png: mean(|SHAP value|) bar chart.
  6. In Top 10 feature có |SHAP| trung bình cao nhất ra console.

Chạy: python notebooks/shap_analysis.py (từ thư mục source-code/)
"""
import os
import sys

# Ensure source-code root is in sys.path (đồng bộ với train_v3.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend — chỉ lưu PNG, không mở cửa sổ GUI
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from src.utils import set_seed
from src.features_v3 import (
    extract_features_v3,
    compute_author_global_preferred_score,
    FEATURE_COLUMN_NAMES,
)

N_SHAP_SAMPLES = 500
TOP_N_FEATURES = 20
TOP_N_PRINT = 10


def main():
    set_seed(42)
    data_dir = os.path.join(BASE_DIR, "data")
    figures_dir = os.path.join(os.path.dirname(BASE_DIR), "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_emb = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    print(f"Loaded Train: {train_df.shape}, SBERT Embeddings: {train_emb.shape}")

    y = train_df['Label'].values.astype(float)

    # Global (pre-fold) Author Preferred Score — same static lookup as train_v3.py
    author_global_stats = compute_author_global_preferred_score(train_df, y)

    # Rebuild the exact V3 full-train feature matrix (fit on the whole train.csv)
    X_full, _ = extract_features_v3(
        train_df, train_emb, is_train=True, y_train=y, author_global_stats=author_global_stats
    )
    feature_names = list(FEATURE_COLUMN_NAMES)
    assert X_full.shape[1] == len(feature_names), (
        f"Feature matrix has {X_full.shape[1]} cols but FEATURE_COLUMN_NAMES has {len(feature_names)}"
    )
    print(f"Feature matrix: {X_full.shape} ({len(feature_names)} named columns)")

    # Train the Stage 2 (Regression) LightGBM model on the full dataset,
    # matching build_stage2_models()'s LightGBM hyperparameters in train_v3.py.
    lgbm_regressor_full = lgb.LGBMRegressor(
        n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24,
        random_state=42, verbose=-1
    )
    lgbm_regressor_full.fit(X_full, y)
    print("Trained Stage-2 LightGBM Regressor on the full train.csv.")

    # SHAP TreeExplainer on the first 500 samples for speed
    n_samples = min(N_SHAP_SAMPLES, X_full.shape[0])
    X_shap = pd.DataFrame(X_full[:n_samples], columns=feature_names)

    explainer = shap.TreeExplainer(lgbm_regressor_full)
    shap_values = explainer.shap_values(X_shap)
    print(f"Computed SHAP values on {n_samples} samples, shape={np.shape(shap_values)}")

    # ---- Plot 1: Beeswarm summary plot (top 20 features) ----
    plt.figure()
    shap.summary_plot(shap_values, X_shap, max_display=TOP_N_FEATURES, show=False)
    out_summary = os.path.join(figures_dir, "06_shap_summary.png")
    plt.tight_layout()
    plt.savefig(out_summary, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_summary}")

    # ---- Plot 2: Mean |SHAP value| bar chart ----
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=TOP_N_FEATURES, show=False)
    out_bar = os.path.join(figures_dir, "07_shap_bar.png")
    plt.tight_layout()
    plt.savefig(out_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_bar}")

    # ---- Top 10 features by mean(|SHAP|) ----
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    print(f"\n===== TOP {TOP_N_PRINT} FEATURES BY MEAN |SHAP VALUE| =====")
    print(importance_df.head(TOP_N_PRINT).to_string(index=False))


if __name__ == "__main__":
    main()
