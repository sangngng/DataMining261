import os
import sys

# Ensure source-code root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.metric import compute_competition_metric
from src.utils import set_seed, save_submission
from src.features_v3 import extract_features_v3, compute_author_global_preferred_score

# -----------------------------------------------------------------------------
# V3 Pipeline: SBERT + SVD + KMeans + Out-of-fold Author/Venue stats
#              + Two-Stage (Stage 1 Classify / Stage 2 LambdaRank Ranking + Regression)
#              + 3-Model Blending (LightGBM + XGBoost + CatBoost)
# -----------------------------------------------------------------------------

RANK_POWER = 1.2  # Priority = RankScore * P(worthwhile) ** RANK_POWER
RANK_NUDGE_WEIGHT = 0.15  # Max +/-15% re-ranking adjustment from the LambdaRank percentile signal


def build_stage1_models(seed: int = 42):
    """Stage 1 (Filter) binary classifiers: P(label > 1)."""
    models = {
        'lgb': lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24,
            random_state=seed, verbose=-1
        ),
        'xgb': xgb.XGBClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=4,
            random_state=seed, eval_metric='logloss'
        ),
    }
    if CATBOOST_AVAILABLE:
        models['cat'] = CatBoostClassifier(
            iterations=180, learning_rate=0.03, depth=5,
            random_seed=seed, verbose=False
        )
    return models


def build_stage2_models(seed: int = 42):
    """Stage 2 (Rank) regressors predicting continuous priority/rating."""
    models = {
        'lgb': lgb.LGBMRegressor(
            n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24,
            random_state=seed, verbose=-1
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=180, learning_rate=0.03, max_depth=4, random_state=seed
        ),
    }
    if CATBOOST_AVAILABLE:
        models['cat'] = CatBoostRegressor(
            iterations=180, learning_rate=0.03, depth=5,
            random_seed=seed, verbose=False
        )
    return models


def build_lambdarank_model(seed: int = 42):
    """LGBMRanker optimizing NDCG-style LambdaRank loss, matching the competition's
    discounted-gain relevance structure (w_r = 1/log2(r+1))."""
    return lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=180,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=24,
        random_state=seed,
        verbose=-1,
        label_gain=[0, 1, 2, 3, 4],  # gain for integer relevance labels 0..4 (y - 1)
    )


def blend_predictions(preds: dict) -> np.ndarray:
    """Simple equal-weight average across available model predictions."""
    arrs = list(preds.values())
    return np.mean(arrs, axis=0)


def run_v3_pipeline():
    set_seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_emb = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    test_emb = np.load(os.path.join(data_dir, "test_embeddings_minilm.npy"))

    print(f"Loaded Train: {train_df.shape}, SBERT Embeddings: {train_emb.shape}")
    print(f"Loaded Test:  {test_df.shape}, SBERT Embeddings: {test_emb.shape}")
    print(f"CatBoost available: {CATBOOST_AVAILABLE}")

    y = train_df['Label'].values.astype(float)
    y_binary = (y > 1.0).astype(int)
    y_rank_rel = (y - 1).astype(int)  # relevance label in {0,1,2,3,4} for LambdaRank

    # Global (pre-fold) Author Preferred Score: computed ONCE on the entire train.csv
    # before any CV splitting, per explicit project direction. Passed as a static,
    # transform-only lookup into every fold (and the full-train/test-predict step) —
    # never refit per fold.
    author_global_stats = compute_author_global_preferred_score(train_df, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n = len(train_df)
    model_names = ['lgb', 'xgb'] + (['cat'] if CATBOOST_AVAILABLE else [])
    oof_proba = {k: np.zeros(n) for k in model_names}
    oof_reg = {k: np.zeros(n) for k in model_names}
    oof_rank = np.zeros(n)

    fold_scores = []

    print("\n========== RUNNING 5-FOLD CV: V3 PIPELINE (AUTHOR/VENUE OOF + LAMBDARANK BLEND) ==========")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), 1):
        df_tr = train_df.iloc[train_idx].reset_index(drop=True)
        df_val = train_df.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y[train_idx], y[val_idx]
        yb_tr = y_binary[train_idx]
        yrank_tr = y_rank_rel[train_idx]

        emb_tr, emb_val = train_emb[train_idx], train_emb[val_idx]

        # Extract features fitted strictly on train fold (no leakage)
        X_tr, cache = extract_features_v3(df_tr, emb_tr, is_train=True, y_train=y_tr, author_global_stats=author_global_stats)
        X_val, _ = extract_features_v3(
            df_val, emb_val, is_train=False,
            kmeans_model=cache['kmeans'],
            svd_model=cache['svd'],
            pos_centroid=cache['pos_centroid'],
            neg_centroid=cache['neg_centroid'],
            author_venue_encoder=cache['author_venue_encoder'],
            author_global_stats=author_global_stats,
        )

        # ---------------- Stage 1: Classification Ensemble ----------------
        clf_models = build_stage1_models(seed=42)
        for name, model in clf_models.items():
            model.fit(X_tr, yb_tr)
            oof_proba[name][val_idx] = model.predict_proba(X_val)[:, 1]

        # ---------------- Stage 2a: Regression Ensemble (raw rating) ----------------
        reg_models = build_stage2_models(seed=42)
        for name, model in reg_models.items():
            model.fit(X_tr, y_tr)
            oof_reg[name][val_idx] = model.predict(X_val)

        # ---------------- Stage 2b: LambdaRank Ranking Model ----------------
        ranker = build_lambdarank_model(seed=42)
        ranker.fit(X_tr, yrank_tr, group=[len(X_tr)])
        oof_rank[val_idx] = ranker.predict(X_val)

        # Fold-level evaluation (equal-weight ensemble blend)
        # NOTE: The regression ensemble (reg_ens) carries the primary rating-scale signal.
        # The LambdaRank score is only a *light* re-ranking nudge (rank-percentile based,
        # not an absolute-scale blend) so it refines ordering without destroying Q.
        p_fold_proba = blend_predictions({k: v[val_idx] for k, v in oof_proba.items()})
        p_fold_reg = blend_predictions({k: v[val_idx] for k, v in oof_reg.items()})
        rank_vals = oof_rank[val_idx]
        rank_percentile = pd.Series(rank_vals).rank(pct=True).values  # stable 0..1 ordering signal
        combined_score = p_fold_reg * (1.0 + RANK_NUDGE_WEIGHT * (rank_percentile - 0.5))

        priority_fold = combined_score * np.power(p_fold_proba, RANK_POWER)
        p_fold = np.where(p_fold_proba >= 0.40, priority_fold, 0.0)
        p_fold = np.clip(p_fold, 0.0, None)

        sc, q, r = compute_competition_metric(df_val['id'].values, y_val, p_fold)
        fold_scores.append(sc)
        print(f"Fold {fold}: Score = {sc:.5f} (Q = {q:.5f}, R = {r:.5f})")

    print(f"\nMean Fold Score: {np.mean(fold_scores):.5f} (std={np.std(fold_scores):.5f})")

    # ---------------- Overall OOF Evaluation & Threshold Sweep ----------------
    oof_proba_ens = blend_predictions(oof_proba)
    oof_reg_ens = blend_predictions(oof_reg)
    oof_rank_percentile = pd.Series(oof_rank).rank(pct=True).values
    oof_combined_score = oof_reg_ens * (1.0 + RANK_NUDGE_WEIGHT * (oof_rank_percentile - 0.5))

    best_t = 0.40
    best_score = 0.0
    best_q = 0.0
    best_r = 0.0

    print("\n--- Sweeping Threshold T* on Out-Of-Fold Predictions [0.20, 0.70] ---")
    for t in np.linspace(0.20, 0.70, 51):
        priority_oof = oof_combined_score * np.power(oof_proba_ens, RANK_POWER)
        p_oof = np.where(oof_proba_ens >= t, priority_oof, 0.0)
        p_oof = np.clip(p_oof, 0.0, None)
        sc, q, r = compute_competition_metric(train_df['id'].values, y, p_oof)
        if sc > best_score:
            best_score = sc
            best_q = q
            best_r = r
            best_t = t

    print(f"\n=======================================================")
    print(f"🔥 V3 PIPELINE OOF RESULTS:")
    print(f"   Optimal Threshold T* = {best_t:.3f}")
    print(f"   OOF Overall Score    = {best_score:.5f}")
    print(f"   Relevance (Q)        = {best_q:.5f}")
    print(f"   Coverage (R)         = {best_r:.5f}")
    print(f"   Mean Fold Score      = {np.mean(fold_scores):.5f}")
    print(f"   -- V2 Reference --     OOF: 0.61256 | Mean Folds: 0.61601")
    print(f"   Delta OOF vs V2       : {best_score - 0.61256:+.5f}")
    print(f"=======================================================\n")

    # ---------------- Train on Full Dataset and Predict on Test Set ----------------
    print("Training on Full Dataset with Optimal T*...")
    X_full, full_cache = extract_features_v3(train_df, train_emb, is_train=True, y_train=y, author_global_stats=author_global_stats)
    X_test, _ = extract_features_v3(
        test_df, test_emb, is_train=False,
        kmeans_model=full_cache['kmeans'],
        svd_model=full_cache['svd'],
        pos_centroid=full_cache['pos_centroid'],
        neg_centroid=full_cache['neg_centroid'],
        author_venue_encoder=full_cache['author_venue_encoder'],
        author_global_stats=author_global_stats,
    )

    # Stage 1 full fit
    clf_models_full = build_stage1_models(seed=42)
    test_proba_preds = {}
    for name, model in clf_models_full.items():
        model.fit(X_full, y_binary)
        test_proba_preds[name] = model.predict_proba(X_test)[:, 1]
    test_proba = blend_predictions(test_proba_preds)

    # Stage 2a full fit (regression)
    reg_models_full = build_stage2_models(seed=42)
    test_reg_preds = {}
    for name, model in reg_models_full.items():
        model.fit(X_full, y)
        test_reg_preds[name] = model.predict(X_test)
    test_reg = blend_predictions(test_reg_preds)

    # Stage 2b full fit (LambdaRank) — light re-ranking nudge only
    ranker_full = build_lambdarank_model(seed=42)
    ranker_full.fit(X_full, y_rank_rel, group=[len(X_full)])
    test_rank = ranker_full.predict(X_test)
    test_rank_percentile = pd.Series(test_rank).rank(pct=True).values

    test_combined_score = test_reg * (1.0 + RANK_NUDGE_WEIGHT * (test_rank_percentile - 0.5))
    test_priorities_raw = test_combined_score * np.power(test_proba, RANK_POWER)
    test_priorities = np.where(test_proba >= best_t, test_priorities_raw, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    # Save submission v3
    sub_dir = os.path.join(base_dir, "submissions")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v3_lambdarank.csv",
        output_dir=sub_dir
    )

    recom_count = int(np.sum(test_priorities > 0))
    pct = 100.0 * recom_count / len(test_priorities)
    print(f"\nRecommended test papers (priority > 0): {recom_count} / {len(test_priorities)} ({pct:.1f}%)")
    print(f"Filtered out test papers (priority == 0): {len(test_priorities) - recom_count} / {len(test_priorities)}")
    print(f"\n[DONE] Saved new submission to: {out_file}")

    return {
        'best_t': best_t,
        'best_score': best_score,
        'best_q': best_q,
        'best_r': best_r,
        'mean_fold_score': float(np.mean(fold_scores)),
        'submission_file': out_file,
        'recom_count': recom_count,
    }


if __name__ == "__main__":
    run_v3_pipeline()
