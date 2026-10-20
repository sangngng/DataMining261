import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, Tuple

from src.metric import compute_competition_metric
from src.features import FeaturePipeline
from src.models import TwoStageRecommender
from src.utils import set_seed

def run_cross_validation(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Runs a Stratified 5-Fold Cross Validation strictly avoiding leakage:
      - Feature pipeline fitted only on train fold
      - Evaluates Relevance (Q), Coverage (R), and Harmonic Mean (Score)
      - Sweeps threshold T on Out-Of-Fold predictions to optimize score
    """
    set_seed(seed)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_raw = train_df['Label'].values
    
    oof_y = np.zeros(len(train_df))
    oof_ids = train_df['id'].values
    oof_proba = np.zeros(len(train_df))
    oof_ratings = np.zeros(len(train_df))

    fold_scores = []

    print(f"\n===== Running {n_splits}-Fold Stratified Cross Validation =====")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_raw), 1):
        df_tr, df_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        y_tr, y_val = y_raw[train_idx], y_raw[val_idx]

        # 1. Feature extraction strictly fitted on train fold
        fe = FeaturePipeline(tfidf_max_features=2500, n_clusters=8, random_state=seed)
        X_tr = fe.fit_transform(df_tr)
        X_val = fe.transform(df_val)

        # 2. Model training
        model = TwoStageRecommender(random_state=seed)
        model.fit(X_tr, y_tr)

        # Save OOF predictions
        oof_y[val_idx] = y_val
        oof_proba[val_idx] = model.clf.predict_proba(X_val)[:, 1]
        oof_ratings[val_idx] = model.reg.predict(X_val)

        # Quick check with default threshold
        p_val = model.predict_priority(X_val, threshold=0.40)
        score_val, q_val, r_val = compute_competition_metric(df_val['id'].values, y_val, p_val)
        fold_scores.append(score_val)
        print(f"Fold {fold}: Score = {score_val:.5f} | Relevance (Q) = {q_val:.5f} | Coverage (R) = {r_val:.5f}")

    print(f"\nMean Fold Score (threshold=0.40): {np.mean(fold_scores):.5f}")

    # Threshold optimization on Out-Of-Fold
    best_t = 0.40
    best_score = 0.0
    best_q = 0.0
    best_r = 0.0

    print("\n--- Sweeping Threshold T on Out-Of-Fold Predictions ---")
    for t in np.linspace(0.10, 0.85, 31):
        p_oof = np.where(oof_proba >= t, oof_ratings, 0.0)
        p_oof = np.clip(p_oof, 0.0, None)
        score_oof, q_oof, r_oof = compute_competition_metric(oof_ids, oof_y, p_oof)
        if score_oof > best_score:
            best_score = score_oof
            best_q = q_oof
            best_r = r_oof
            best_t = t

    print(f"\n Optimal Threshold T* = {best_t:.2f}")
    print(f" Best OOF Overall Score = {best_score:.5f}")
    print(f"   - Relevance (Q)      = {best_q:.5f}")
    print(f"   - Coverage (R)       = {best_r:.5f}")
    
    return best_score, best_q, best_r, best_t
