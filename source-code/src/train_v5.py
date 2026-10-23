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


PROCEEDINGS_KEYWORDS = [
    'proceedings', 'conference on', 'preface', 'workshop on',
    'doctoral consortium', 'front matter', 'editorial'
]


def extract_features_v5(
    df: pd.DataFrame,
    emb_minilm: np.ndarray,
    emb_bge: np.ndarray,
    is_train: bool = True,
    kmeans_model=None,
    svd_model=None,
    pos_centroid_mini=None,
    top5_centroid_mini=None,
    neg_centroid_mini=None,
    pos_centroid_bge=None,
    top5_centroid_bge=None,
    neg_centroid_bge=None,
    venue_stats=None,
    y_train=None,
):
    """
    V5 Dual-Embedding Feature Pipeline:
    Combines all-MiniLM-L6-v2 (384d) + BAAI/bge-small-en-v1.5 (384d) into 768d dual representation.
    Extracts multi-centroid similarities on both embeddings, SVD(64), KMeans(12),
    proceedings hard filter flag, and Bayesian venue target encoding.
    """
    n_samples = len(df)

    # 1. Multi-Centroid Cosine Similarities on MiniLM
    if is_train:
        pos_mask = (y_train >= 4)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        pos_centroid_mini = emb_minilm[pos_mask].mean(axis=0, keepdims=True)
        top5_centroid_mini = emb_minilm[top5_mask].mean(axis=0, keepdims=True)
        neg_centroid_mini = emb_minilm[neg_mask].mean(axis=0, keepdims=True)

        pos_centroid_mini /= (np.linalg.norm(pos_centroid_mini) + 1e-9)
        top5_centroid_mini /= (np.linalg.norm(top5_centroid_mini) + 1e-9)
        neg_centroid_mini /= (np.linalg.norm(neg_centroid_mini) + 1e-9)

        pos_centroid_bge = emb_bge[pos_mask].mean(axis=0, keepdims=True)
        top5_centroid_bge = emb_bge[top5_mask].mean(axis=0, keepdims=True)
        neg_centroid_bge = emb_bge[neg_mask].mean(axis=0, keepdims=True)

        pos_centroid_bge /= (np.linalg.norm(pos_centroid_bge) + 1e-9)
        top5_centroid_bge /= (np.linalg.norm(top5_centroid_bge) + 1e-9)
        neg_centroid_bge /= (np.linalg.norm(neg_centroid_bge) + 1e-9)

    # MiniLM similarities
    sim_pos_m = np.dot(emb_minilm, pos_centroid_mini.T)
    sim_top5_m = np.dot(emb_minilm, top5_centroid_mini.T)
    sim_neg_m = np.dot(emb_minilm, neg_centroid_mini.T)
    sim_diff_m = sim_pos_m - sim_neg_m

    # BGE similarities
    sim_pos_b = np.dot(emb_bge, pos_centroid_bge.T)
    sim_top5_b = np.dot(emb_bge, top5_centroid_bge.T)
    sim_neg_b = np.dot(emb_bge, neg_centroid_bge.T)
    sim_diff_b = sim_pos_b - sim_neg_b

    # Blended similarity
    sim_diff_avg = 0.5 * (sim_diff_m + sim_diff_b)
    sim_top5_avg = 0.5 * (sim_top5_m + sim_top5_b)

    # 2. Dual Embeddings Concatenation (768d) & SVD (top 64 latent axes)
    dual_emb = np.hstack([emb_minilm, emb_bge])
    if is_train:
        svd_model = TruncatedSVD(n_components=64, random_state=42)
        svd_feats = svd_model.fit_transform(dual_emb)
    else:
        svd_feats = svd_model.transform(dual_emb)

    # 3. K-Means Semantic Clustering on Dual Embeddings (K=12)
    if is_train:
        kmeans_model = KMeans(n_clusters=12, random_state=42, n_init=10)
        cluster_dists = kmeans_model.fit_transform(dual_emb)
    else:
        cluster_dists = kmeans_model.transform(dual_emb)

    # 4. Clean Metadata & Proceedings Hard Flag
    title_series = df['title'].fillna('')
    pat = '|'.join(PROCEEDINGS_KEYWORDS)
    is_proceedings = title_series.str.contains(pat, case=False, na=False).astype(float).values.reshape(-1, 1)

    year_num = pd.to_numeric(df['year'], errors='coerce').fillna(2020.0).values.reshape(-1, 1)
    year_norm = (year_num - 2000.0) / 26.0
    paper_age = (2026.0 - year_num) / 10.0
    is_recent = (year_num >= 2024.0).astype(float)
    has_doi = (~df['doi'].isna()).astype(float).values.reshape(-1, 1)

    author_series = df['authors'].fillna('')
    author_count = author_series.apply(lambda s: len(s.split(',')) if len(s.strip()) > 0 else 0).values.reshape(-1, 1)
    is_single_author = (author_count == 1).astype(float)

    char_len = (title_series.apply(len).values.reshape(-1, 1) - 50.0) / 50.0
    word_count = (title_series.apply(lambda s: len(s.split())).values.reshape(-1, 1) - 10.0) / 10.0

    # Interaction terms
    sim_recent_inter = sim_diff_avg * is_recent
    sim_doi_inter = sim_diff_avg * has_doi

    # 5. Bayesian Smoothed Venue Target Encoding (m=5.0)
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

    # Assemble all features (64 + 12 + 10 + 10 = 96 columns)
    X = np.hstack([
        svd_feats,          # 64
        cluster_dists,      # 12
        sim_pos_m,          # 1
        sim_top5_m,         # 1
        sim_neg_m,          # 1
        sim_diff_m,         # 1
        sim_pos_b,          # 1
        sim_top5_b,         # 1
        sim_neg_b,          # 1
        sim_diff_b,         # 1
        sim_diff_avg,       # 1
        sim_top5_avg,       # 1
        is_proceedings,     # 1 (Hard filter signal)
        year_norm,          # 1
        paper_age,          # 1
        is_recent,          # 1
        has_doi,            # 1
        author_count,       # 1
        is_single_author,   # 1
        char_len,           # 1
        word_count,         # 1
        sim_recent_inter,   # 1
        sim_doi_inter,      # 1
        v_smoothed,         # 1
        v_freq,             # 1
    ])

    cache = {
        'kmeans': kmeans_model,
        'svd': svd_model,
        'pos_centroid_mini': pos_centroid_mini,
        'top5_centroid_mini': top5_centroid_mini,
        'neg_centroid_mini': neg_centroid_mini,
        'pos_centroid_bge': pos_centroid_bge,
        'top5_centroid_bge': top5_centroid_bge,
        'neg_centroid_bge': neg_centroid_bge,
        'venue_stats': venue_stats,
    }
    return X, is_proceedings.flatten(), cache


def run_v5_pipeline():
    set_seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_emb_mini = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    test_emb_mini = np.load(os.path.join(data_dir, "test_embeddings_minilm.npy"))

    train_emb_bge = np.load(os.path.join(data_dir, "train_embeddings_bge.npy"))
    test_emb_bge = np.load(os.path.join(data_dir, "test_embeddings_bge.npy"))

    print(f"Loaded Train: {train_df.shape}, MiniLM: {train_emb_mini.shape}, BGE: {train_emb_bge.shape}")
    print(f"Loaded Test:  {test_df.shape}, MiniLM: {test_emb_mini.shape}, BGE: {test_emb_bge.shape}")

    y = train_df['Label'].values
    y_binary = (y > 1.0).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n = len(train_df)
    oof_proba = np.zeros(n)
    oof_reg = np.zeros(n)

    fold_scores = []
    print("\n--- Starting 5-Fold Stratified CV (V5 Dual-Embedding Tri-Model Ensemble) ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_binary)):
        df_tr, df_val = train_df.iloc[train_idx].copy(), train_df.iloc[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]
        y_tr_bin, y_val_bin = y_binary[train_idx], y_binary[val_idx]

        mini_tr, mini_val = train_emb_mini[train_idx], train_emb_mini[val_idx]
        bge_tr, bge_val = train_emb_bge[train_idx], train_emb_bge[val_idx]

        X_tr, proc_tr, cache = extract_features_v5(
            df_tr, mini_tr, bge_tr, is_train=True, y_train=y_tr
        )
        X_val, proc_val, _ = extract_features_v5(
            df_val, mini_val, bge_val, is_train=False,
            kmeans_model=cache['kmeans'],
            svd_model=cache['svd'],
            pos_centroid_mini=cache['pos_centroid_mini'],
            top5_centroid_mini=cache['top5_centroid_mini'],
            neg_centroid_mini=cache['neg_centroid_mini'],
            pos_centroid_bge=cache['pos_centroid_bge'],
            top5_centroid_bge=cache['top5_centroid_bge'],
            neg_centroid_bge=cache['neg_centroid_bge'],
            venue_stats=cache['venue_stats'],
        )

        # ---------------- Stage 1: Classification Ensemble ----------------
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=280, learning_rate=0.02, max_depth=5, num_leaves=24,
            subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
        )
        clf_lgb.fit(X_tr, y_tr_bin)
        p_val_clf_lgb = clf_lgb.predict_proba(X_val)[:, 1]

        clf_xgb = xgb.XGBClassifier(
            n_estimators=260, learning_rate=0.02, max_depth=4,
            subsample=0.85, colsample_bytree=0.75, random_state=42, eval_metric='logloss'
        )
        clf_xgb.fit(X_tr, y_tr_bin)
        p_val_clf_xgb = clf_xgb.predict_proba(X_val)[:, 1]

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

        # Force proceedings to near-zero
        p_val_proba[proc_val == 1.0] = 0.0
        oof_proba[val_idx] = p_val_proba

        # ---------------- Stage 2: Regression Ensemble ----------------
        reg_lgb = lgb.LGBMRegressor(
            n_estimators=280, learning_rate=0.02, max_depth=5, num_leaves=24,
            subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
        )
        reg_lgb.fit(X_tr, y_tr)
        p_val_reg_lgb = reg_lgb.predict(X_val)

        reg_xgb = xgb.XGBRegressor(
            n_estimators=260, learning_rate=0.02, max_depth=4,
            subsample=0.85, colsample_bytree=0.75, random_state=42
        )
        reg_xgb.fit(X_tr, y_tr)
        p_val_reg_xgb = reg_xgb.predict(X_val)

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
        fold_priorities = np.where(p_val_proba >= 0.525, p_val_reg, 0.0)
        f_score, f_q, f_r = compute_competition_metric(df_val['id'].values, y_val, fold_priorities)
        fold_scores.append(f_score)
        print(f"Fold {fold+1} Score: {f_score:.5f} (Q={f_q:.5f}, R={f_r:.5f})")

    # ---------------- Full Dataset Feature Extraction ----------------
    print("\nExtracting features on full train and test sets (Dual Embeddings)...")
    X_full, proc_full, full_cache = extract_features_v5(
        train_df, train_emb_mini, train_emb_bge, is_train=True, y_train=y
    )
    X_test, proc_test, _ = extract_features_v5(
        test_df, test_emb_mini, test_emb_bge, is_train=False,
        kmeans_model=full_cache['kmeans'],
        svd_model=full_cache['svd'],
        pos_centroid_mini=full_cache['pos_centroid_mini'],
        top5_centroid_mini=full_cache['top5_centroid_mini'],
        neg_centroid_mini=full_cache['neg_centroid_mini'],
        pos_centroid_bge=full_cache['pos_centroid_bge'],
        top5_centroid_bge=full_cache['top5_centroid_bge'],
        neg_centroid_bge=full_cache['neg_centroid_bge'],
        venue_stats=full_cache['venue_stats'],
    )

    # Full Stage 1 Models
    print("Fitting Full Classification Ensemble...")
    clf_lgb_full = lgb.LGBMClassifier(
        n_estimators=280, learning_rate=0.02, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
    )
    clf_lgb_full.fit(X_full, y_binary)

    clf_xgb_full = xgb.XGBClassifier(
        n_estimators=260, learning_rate=0.02, max_depth=4,
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

    # Hard filter proceedings on test set
    test_proba[proc_test == 1.0] = 0.0

    # Full Stage 2 Models
    print("Fitting Full Regression Ensemble...")
    reg_lgb_full = lgb.LGBMRegressor(
        n_estimators=280, learning_rate=0.02, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1
    )
    reg_lgb_full.fit(X_full, y)

    reg_xgb_full = xgb.XGBRegressor(
        n_estimators=260, learning_rate=0.02, max_depth=4,
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
    print("\n--- Sweeping Threshold T* Constrained to Golden Zone [368, 382] ---")
    best_score = -1.0
    best_t = 0.525
    best_q = 0.0
    best_r = 0.0
    best_test_count = 0

    threshold_candidates = np.linspace(0.40, 0.65, 101)

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

    print(f"\n[OPTIMAL CALIBRATION FOUND FOR V5]")
    print(f"Optimal Threshold T*: {best_t:.3f}")
    print(f"OOF Overall Score:   {best_score:.5f} (Q={best_q:.5f}, R={best_r:.5f})")
    print(f"Mean Fold Score:     {np.mean(fold_scores):.5f} (std={np.std(fold_scores):.5f})")
    print(f"Test Recommendations: {best_test_count} / {len(test_df)} ({best_test_count/len(test_df)*100:.1f}%)")

    # Final Test Priorities (Pure Rating Scale)
    test_priorities = np.where(test_proba >= best_t, test_reg, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    # Save submission v5
    sub_dir = os.path.join(base_dir, "submissions")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v5_dual_embedding.csv",
        output_dir=sub_dir
    )

    print(f"\n[SUCCESS] Saved V5 submission to: {out_file}")
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
    run_v5_pipeline()
