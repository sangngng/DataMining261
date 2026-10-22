import os
import sys

# Ensure source-code root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.metric import compute_competition_metric
from src.utils import set_seed, save_submission


def extract_features_v4(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    is_train: bool = True,
    kmeans_model=None,
    svd_model=None,
    pos_centroid=None,
    top5_centroid=None,
    neg_centroid=None,
    venue_stats=None,
    y_train=None,
):
    """
    V4 Clean Semantic Features: Strictly avoids author leakage and distribution shift.
    Inherits the proven V2 semantic core: SBERT (384d) + SVD(48) + KMeans(12) +
    Preference Centroids + Bayesian Smoothed Venue + Clean Metadata.
    """
    n_samples = len(df)

    # 1. Multi-Centroid Preference Cosine Similarities
    if is_train:
        pos_mask = (y_train >= 4)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        pos_centroid = embeddings[pos_mask].mean(axis=0, keepdims=True)
        top5_centroid = embeddings[top5_mask].mean(axis=0, keepdims=True)
        neg_centroid = embeddings[neg_mask].mean(axis=0, keepdims=True)

        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-9)
        top5_centroid = top5_centroid / (np.linalg.norm(top5_centroid) + 1e-9)
        neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-9)

    sim_pos = np.dot(embeddings, pos_centroid.T)    # (N, 1)
    sim_top5 = np.dot(embeddings, top5_centroid.T)  # (N, 1)
    sim_neg = np.dot(embeddings, neg_centroid.T)    # (N, 1)
    sim_diff = sim_pos - sim_neg                    # (N, 1) - Key feature #1 from SHAP
    sim_diff_top5 = sim_top5 - sim_neg              # (N, 1)

    # 2. SVD Dimensionality Reduction on SBERT (48 latent semantic axes)
    if is_train:
        svd_model = TruncatedSVD(n_components=48, random_state=42)
        svd_feats = svd_model.fit_transform(embeddings)
    else:
        svd_feats = svd_model.transform(embeddings)

    # 3. K-Means Semantic Topic Clustering (K=12)
    if is_train:
        kmeans_model = KMeans(n_clusters=12, random_state=42, n_init=10)
        cluster_dists = kmeans_model.fit_transform(embeddings)
    else:
        cluster_dists = kmeans_model.transform(embeddings)

    # 4. Clean Metadata Features
    year_num = pd.to_numeric(df['year'], errors='coerce').fillna(2020.0).values.reshape(-1, 1)
    year_norm = (year_num - 2000.0) / 26.0
    paper_age = (2026.0 - year_num) / 10.0
    is_recent = (year_num >= 2024.0).astype(float)

    has_doi = (~df['doi'].isna()).astype(float).values.reshape(-1, 1)

    # Author structural counts only (NO target encoding / label history)
    author_series = df['authors'].fillna('')
    author_count = author_series.apply(lambda s: len(s.split(',')) if len(s.strip()) > 0 else 0).values.reshape(-1, 1)
    is_single_author = (author_count == 1).astype(float)

    # Title length
    title_series = df['title'].fillna('')
    char_len = (title_series.apply(len).values.reshape(-1, 1) - 50.0) / 50.0
    word_count = (title_series.apply(lambda s: len(s.split())).values.reshape(-1, 1) - 10.0) / 10.0

    # Interaction terms
    sim_recent_inter = sim_diff * is_recent
    sim_doi_inter = sim_diff * has_doi

    # 5. Bayesian-Smoothed Venue Target Encoding (smoothing m=5.0)
    venue_series = df['venue'].fillna('UNKNOWN').astype(str)
    if is_train:
        m = 5.0
        global_mean = float(y_train.mean())
        venue_stats_dict = {}
        grouped = df.groupby(venue_series)['Label'].agg(['count', 'mean'])
        for v_name, row in grouped.iterrows():
            cnt = row['count']
            raw_m = row['mean']
            smoothed = (cnt * raw_m + m * global_mean) / (cnt + m)
            venue_stats_dict[v_name] = (smoothed, cnt)
        venue_counts = venue_series.value_counts().to_dict()
        venue_stats = {
            'smoothed_map': venue_stats_dict,
            'counts': venue_counts,
            'global_mean': global_mean
        }

    s_map = venue_stats['smoothed_map']
    c_map = venue_stats['counts']
    g_mean = venue_stats['global_mean']

    v_smoothed = np.array([s_map[v][0] if v in s_map else g_mean for v in venue_series]).reshape(-1, 1)
    v_freq = np.log1p(np.array([c_map.get(v, 0.0) for v in venue_series])).reshape(-1, 1)

    # Assemble all features
    X = np.hstack([
        svd_feats,           # 48
        cluster_dists,       # 12
        sim_pos,             # 1
        sim_top5,            # 1
        sim_neg,             # 1
        sim_diff,            # 1
        sim_diff_top5,       # 1
        year_norm,           # 1
        paper_age,           # 1
        is_recent,           # 1
        has_doi,             # 1
        author_count,        # 1
        is_single_author,    # 1
        char_len,            # 1
        word_count,          # 1
        sim_recent_inter,    # 1
        sim_doi_inter,       # 1
        v_smoothed,          # 1
        v_freq,              # 1
    ])

    cache = {
        'kmeans': kmeans_model,
        'svd': svd_model,
        'pos_centroid': pos_centroid,
        'top5_centroid': top5_centroid,
        'neg_centroid': neg_centroid,
        'venue_stats': venue_stats,
    }
    return X, cache


def run_v4_pipeline():
    set_seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_emb = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    test_emb = np.load(os.path.join(data_dir, "test_embeddings_minilm.npy"))

    print(f"Loaded Train: {train_df.shape}, SBERT Embeddings: {train_emb.shape}")
    print(f"Loaded Test:  {test_df.shape}, SBERT Embeddings: {test_emb.shape}")

    y = train_df['Label'].values
    y_binary = (y > 1.0).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n = len(train_df)
    oof_proba = np.zeros(n)
    oof_reg = np.zeros(n)

    fold_scores = []
    print("\n--- Starting 5-Fold Stratified Cross-Validation (V4 Clean Semantic Ensemble) ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_binary)):
        df_tr, df_val = train_df.iloc[train_idx].copy(), train_df.iloc[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]
        y_tr_bin, y_val_bin = y_binary[train_idx], y_binary[val_idx]

        emb_tr, emb_val = train_emb[train_idx], train_emb[val_idx]

        X_tr, cache = extract_features_v4(df_tr, emb_tr, is_train=True, y_train=y_tr)
        X_val, _ = extract_features_v4(
            df_val, emb_val, is_train=False,
            kmeans_model=cache['kmeans'],
            svd_model=cache['svd'],
            pos_centroid=cache['pos_centroid'],
            top5_centroid=cache['top5_centroid'],
            neg_centroid=cache['neg_centroid'],
            venue_stats=cache['venue_stats'],
        )

        # ---------------- Stage 1: Classification Ensemble ----------------
        # 1. LightGBM Classifier
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=260, learning_rate=0.02, max_depth=5, num_leaves=24,
            subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
        )
        clf_lgb.fit(X_tr, y_tr_bin)
        p_val_clf_lgb = clf_lgb.predict_proba(X_val)[:, 1]

        # 2. XGBoost Classifier
        clf_xgb = xgb.XGBClassifier(
            n_estimators=240, learning_rate=0.02, max_depth=4,
            subsample=0.85, colsample_bytree=0.75, random_state=42, eval_metric='logloss'
        )
        clf_xgb.fit(X_tr, y_tr_bin)
        p_val_clf_xgb = clf_xgb.predict_proba(X_val)[:, 1]

        # 3. CatBoost Classifier
        if CATBOOST_AVAILABLE:
            clf_cat = CatBoostClassifier(
                iterations=260, learning_rate=0.025, depth=4,
                random_seed=42, verbose=False
            )
            clf_cat.fit(X_tr, y_tr_bin)
            p_val_clf_cat = clf_cat.predict_proba(X_val)[:, 1]
            p_val_proba = 0.40 * p_val_clf_lgb + 0.40 * p_val_clf_xgb + 0.20 * p_val_clf_cat
        else:
            p_val_proba = 0.50 * p_val_clf_lgb + 0.50 * p_val_clf_xgb

        oof_proba[val_idx] = p_val_proba

        # ---------------- Stage 2: Regression Ensemble ----------------
        # 1. LightGBM Regressor
        reg_lgb = lgb.LGBMRegressor(
            n_estimators=260, learning_rate=0.02, max_depth=5, num_leaves=24,
            subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
        )
        reg_lgb.fit(X_tr, y_tr)
        p_val_reg_lgb = reg_lgb.predict(X_val)

        # 2. XGBoost Regressor
        reg_xgb = xgb.XGBRegressor(
            n_estimators=240, learning_rate=0.02, max_depth=4,
            subsample=0.85, colsample_bytree=0.75, random_state=42
        )
        reg_xgb.fit(X_tr, y_tr)
        p_val_reg_xgb = reg_xgb.predict(X_val)

        # 3. CatBoost Regressor
        if CATBOOST_AVAILABLE:
            reg_cat = CatBoostRegressor(
                iterations=260, learning_rate=0.025, depth=4,
                random_seed=42, verbose=False
            )
            reg_cat.fit(X_tr, y_tr)
            p_val_reg_cat = reg_cat.predict(X_val)
            p_val_reg = 0.40 * p_val_reg_lgb + 0.40 * p_val_reg_xgb + 0.20 * p_val_reg_cat
        else:
            p_val_reg = 0.50 * p_val_reg_lgb + 0.50 * p_val_reg_xgb

        oof_reg[val_idx] = p_val_reg

        # Fold local evaluation
        fold_priorities = np.where(p_val_proba >= 0.53, p_val_reg, 0.0)
        f_score, f_q, f_r = compute_competition_metric(df_val['id'].values, y_val, fold_priorities)
        fold_scores.append(f_score)
        print(f"Fold {fold+1} Score: {f_score:.5f} (Q={f_q:.5f}, R={f_r:.5f})")

    # ---------------- Full Dataset Feature Extraction ----------------
    print("\nExtracting features on full train and test sets...")
    X_full, full_cache = extract_features_v4(train_df, train_emb, is_train=True, y_train=y)
    X_test, _ = extract_features_v4(
        test_df, test_emb, is_train=False,
        kmeans_model=full_cache['kmeans'],
        svd_model=full_cache['svd'],
        pos_centroid=full_cache['pos_centroid'],
        top5_centroid=full_cache['top5_centroid'],
        neg_centroid=full_cache['neg_centroid'],
        venue_stats=full_cache['venue_stats'],
    )

    # Full Stage 1 Models
    print("Fitting Full Classification Ensemble...")
    clf_lgb_full = lgb.LGBMClassifier(
        n_estimators=260, learning_rate=0.02, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
    )
    clf_lgb_full.fit(X_full, y_binary)

    clf_xgb_full = xgb.XGBClassifier(
        n_estimators=240, learning_rate=0.02, max_depth=4,
        subsample=0.85, colsample_bytree=0.75, random_state=42, eval_metric='logloss'
    )
    clf_xgb_full.fit(X_full, y_binary)

    if CATBOOST_AVAILABLE:
        clf_cat_full = CatBoostClassifier(iterations=260, learning_rate=0.025, depth=4, random_seed=42, verbose=False)
        clf_cat_full.fit(X_full, y_binary)
        test_proba = (
            0.40 * clf_lgb_full.predict_proba(X_test)[:, 1] +
            0.40 * clf_xgb_full.predict_proba(X_test)[:, 1] +
            0.20 * clf_cat_full.predict_proba(X_test)[:, 1]
        )
    else:
        test_proba = (
            0.50 * clf_lgb_full.predict_proba(X_test)[:, 1] +
            0.50 * clf_xgb_full.predict_proba(X_test)[:, 1]
        )

    # Full Stage 2 Models
    print("Fitting Full Regression Ensemble...")
    reg_lgb_full = lgb.LGBMRegressor(
        n_estimators=260, learning_rate=0.02, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
    )
    reg_lgb_full.fit(X_full, y)

    reg_xgb_full = xgb.XGBRegressor(
        n_estimators=240, learning_rate=0.02, max_depth=4,
        subsample=0.85, colsample_bytree=0.75, random_state=42
    )
    reg_xgb_full.fit(X_full, y)

    if CATBOOST_AVAILABLE:
        reg_cat_full = CatBoostRegressor(iterations=260, learning_rate=0.025, depth=4, random_seed=42, verbose=False)
        reg_cat_full.fit(X_full, y)
        test_reg = (
            0.40 * reg_lgb_full.predict(X_test) +
            0.40 * reg_xgb_full.predict(X_test) +
            0.20 * reg_cat_full.predict(X_test)
        )
    else:
        test_reg = (
            0.50 * reg_lgb_full.predict(X_test) +
            0.50 * reg_xgb_full.predict(X_test)
        )

    # ---------------- Calibrate Threshold T* in Golden Zone [368, 382] ----------------
    print("\n--- Sweeping Threshold T* Constrained to Golden Zone [368, 382] (61.7% - 64.1%) ---")
    best_score = -1.0
    best_t = 0.53
    best_q = 0.0
    best_r = 0.0
    best_test_count = 0

    threshold_candidates = np.linspace(0.40, 0.65, 101)

    # Pass 1: Within Golden Zone
    golden_min, golden_max = 368, 382
    for t in threshold_candidates:
        test_count = int(np.sum(test_proba >= t))
        if golden_min <= test_count <= golden_max:
            priorities_oof = np.where(oof_proba >= t, oof_reg, 0.0)
            score, q, r = compute_competition_metric(train_df['id'].values, y, priorities_oof)
            if score > best_score:
                best_score = score
                best_t = float(t)
                best_q = q
                best_r = r
                best_test_count = test_count

    # Fallback if no exact threshold found in golden zone
    if best_score < 0:
        print("[WARNING] No threshold inside exact golden zone; selecting closest.")
        best_diff = 9999
        for t in threshold_candidates:
            test_count = int(np.sum(test_proba >= t))
            diff = abs(test_count - 375)
            priorities_oof = np.where(oof_proba >= t, oof_reg, 0.0)
            score, q, r = compute_competition_metric(train_df['id'].values, y, priorities_oof)
            if diff < best_diff or (diff == best_diff and score > best_score):
                best_diff = diff
                best_score = score
                best_t = float(t)
                best_q = q
                best_r = r
                best_test_count = test_count

    print(f"\n[OPTIMAL CALIBRATION FOUND]")
    print(f"Optimal Threshold T*: {best_t:.3f}")
    print(f"OOF Overall Score:   {best_score:.5f} (Q={best_q:.5f}, R={best_r:.5f})")
    print(f"Mean Fold Score:     {np.mean(fold_scores):.5f} (std={np.std(fold_scores):.5f})")
    print(f"Test Recommendations: {best_test_count} / {len(test_df)} ({best_test_count/len(test_df)*100:.1f}%)")

    # Final Test Priorities (Pure Rating Scale)
    test_priorities = np.where(test_proba >= best_t, test_reg, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    # Save submission v4
    sub_dir = os.path.join(base_dir, "submissions")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v4_clean_ensemble.csv",
        output_dir=sub_dir
    )

    print(f"\n[SUCCESS] Saved V4 submission to: {out_file}")
    return {
        'best_t': best_t,
        'best_score': best_score,
        'best_q': best_q,
        'best_r': best_r,
        'mean_fold_score': float(np.mean(fold_scores)),
        'test_count': best_test_count,
        'submission_file': out_file
    }


if __name__ == "__main__":
    run_v4_pipeline()
