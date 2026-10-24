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


def extract_features_v9(
    df: pd.DataFrame,
    emb_mpnet: np.ndarray,
    emb_bge: np.ndarray,
    is_train: bool = True,
    kmeans_model=None,
    svd_model=None,
    pos_c_mpnet=None,
    top5_c_mpnet=None,
    neg_c_mpnet=None,
    pos_c_bge=None,
    top5_c_bge=None,
    neg_c_bge=None,
    sub_c_pos=None,
    sub_c_neg=None,
    venue_stats=None,
    y_train=None,
):
    """
    V9 Deep Dense Frontier Pipeline:
    Combines all-mpnet-base-v2 (768d) + BAAI/bge-small-en-v1.5 (384d) = 1152d dense representation.
    Extracts global centroids, multi-modal sub-centroids (5 positive clusters + 3 negative clusters),
    SVD(64), KMeans(12), proceedings hard filter, and Bayesian venue target encoding.
    STRICTLY ZERO author rating leakage to protect Public and Private Leaderboard generalizability.
    """
    n_samples = len(df)
    dense_1152 = np.hstack([emb_mpnet, emb_bge])

    # 1. Global Centroids on MPNet & BGE
    if is_train:
        pos_mask = (y_train >= 4)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        # MPNet centroids
        pos_c_mpnet = emb_mpnet[pos_mask].mean(axis=0, keepdims=True)
        top5_c_mpnet = emb_mpnet[top5_mask].mean(axis=0, keepdims=True)
        neg_c_mpnet = emb_mpnet[neg_mask].mean(axis=0, keepdims=True)
        pos_c_mpnet /= (np.linalg.norm(pos_c_mpnet) + 1e-9)
        top5_c_mpnet /= (np.linalg.norm(top5_c_mpnet) + 1e-9)
        neg_c_mpnet /= (np.linalg.norm(neg_c_mpnet) + 1e-9)

        # BGE centroids
        pos_c_bge = emb_bge[pos_mask].mean(axis=0, keepdims=True)
        top5_c_bge = emb_bge[top5_mask].mean(axis=0, keepdims=True)
        neg_c_bge = emb_bge[neg_mask].mean(axis=0, keepdims=True)
        pos_c_bge /= (np.linalg.norm(pos_c_bge) + 1e-9)
        top5_c_bge /= (np.linalg.norm(top5_c_bge) + 1e-9)
        neg_c_bge /= (np.linalg.norm(neg_c_bge) + 1e-9)

        # Multi-modal Sub-Centroids (K=5 positive sub-topics on 1152d)
        km_pos = KMeans(n_clusters=5, random_state=42, n_init=5)
        km_pos.fit(dense_1152[pos_mask])
        sub_c_pos = km_pos.cluster_centers_
        sub_c_pos /= (np.linalg.norm(sub_c_pos, axis=1, keepdims=True) + 1e-9)

        # Multi-modal Sub-Centroids (K=3 negative sub-topics on 1152d)
        km_neg = KMeans(n_clusters=3, random_state=42, n_init=5)
        km_neg.fit(dense_1152[neg_mask])
        sub_c_neg = km_neg.cluster_centers_
        sub_c_neg /= (np.linalg.norm(sub_c_neg, axis=1, keepdims=True) + 1e-9)

    # Similarities to MPNet global centroids
    sim_pos_m = np.dot(emb_mpnet, pos_c_mpnet.T)
    sim_top5_m = np.dot(emb_mpnet, top5_c_mpnet.T)
    sim_neg_m = np.dot(emb_mpnet, neg_c_mpnet.T)
    sim_diff_m = sim_pos_m - sim_neg_m

    # Similarities to BGE global centroids
    sim_pos_b = np.dot(emb_bge, pos_c_bge.T)
    sim_top5_b = np.dot(emb_bge, top5_c_bge.T)
    sim_neg_b = np.dot(emb_bge, neg_c_bge.T)
    sim_diff_b = sim_pos_b - sim_neg_b

    # Blended global similarities
    sim_diff_avg = 0.5 * (sim_diff_m + sim_diff_b)
    sim_top5_avg = 0.5 * (sim_top5_m + sim_top5_b)

    # Sub-Centroid Similarities on 1152d
    dense_norm = dense_1152 / (np.linalg.norm(dense_1152, axis=1, keepdims=True) + 1e-9)
    sub_sim_pos = np.dot(dense_norm, sub_c_pos.T) # (N, 5)
    max_sub_sim = np.max(sub_sim_pos, axis=1, keepdims=True)
    mean_sub_sim = np.mean(sub_sim_pos, axis=1, keepdims=True)

    sub_sim_neg = np.dot(dense_norm, sub_c_neg.T) # (N, 3)
    max_neg_sub_sim = np.max(sub_sim_neg, axis=1, keepdims=True)

    margin_sub = max_sub_sim - max_neg_sub_sim

    # 2. SVD (64 components) on 1152d
    if is_train:
        svd_model = TruncatedSVD(n_components=64, random_state=42)
        svd_feats = svd_model.fit_transform(dense_1152)
    else:
        svd_feats = svd_model.transform(dense_1152)

    # 3. KMeans (12 topic clusters) on 1152d
    if is_train:
        kmeans_model = KMeans(n_clusters=12, random_state=42, n_init=10)
        cluster_dists = kmeans_model.fit_transform(dense_1152)
    else:
        cluster_dists = kmeans_model.transform(dense_1152)

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

    # Interactions
    sim_recent_inter = sim_diff_avg * is_recent
    sim_doi_inter = sim_diff_avg * has_doi

    # 5. Bayesian Smoothed Venue Target Encoding (m=5.0)
    venue_series = df['venue'].fillna('UNKNOWN').astype(str)
    if is_train:
        m_venue = 5.0
        global_mean = float(y_train.mean())
        venue_stats_dict = {}
        grouped = df.groupby(venue_series)['Label'].agg(['count', 'mean'])
        for v_name, row in grouped.iterrows():
            cnt = row['count']
            raw_m = row['mean']
            smoothed = (cnt * raw_m + m_venue * global_mean) / (cnt + m_venue)
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
        sub_sim_pos,        # 5
        max_sub_sim,        # 1
        mean_sub_sim,       # 1
        sub_sim_neg,        # 3
        max_neg_sub_sim,    # 1
        margin_sub,         # 1
        is_proceedings,     # 1
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
        'pos_c_mpnet': pos_c_mpnet,
        'top5_c_mpnet': top5_c_mpnet,
        'neg_c_mpnet': neg_c_mpnet,
        'pos_c_bge': pos_c_bge,
        'top5_c_bge': top5_c_bge,
        'neg_c_bge': neg_c_bge,
        'sub_c_pos': sub_c_pos,
        'sub_c_neg': sub_c_neg,
        'venue_stats': venue_stats,
    }
    return X, is_proceedings.flatten(), cache


def run_v9_pipeline():
    set_seed(42)
    print("==========================================================================")
    print("🚀 LAUNCHING V9: MPNET-DENSE FRONTIER & MULTI-MODAL SUB-CENTROIDS")
    print("==========================================================================")

    data_dir = os.path.join(BASE_DIR, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Load MPNet and BGE embeddings
    mpnet_train = np.load(os.path.join(data_dir, "train_embeddings_mpnet.npy"))
    mpnet_test = np.load(os.path.join(data_dir, "test_embeddings_mpnet.npy"))
    bge_train = np.load(os.path.join(data_dir, "train_embeddings_bge.npy"))
    bge_test = np.load(os.path.join(data_dir, "test_embeddings_bge.npy"))

    print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}")
    print(f"Embeddings: MPNet={mpnet_train.shape}, BGE={bge_train.shape}")

    y = train_df['Label'].values
    y_binary = (y > 1).astype(int)

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = np.zeros(len(train_df))
    oof_reg = np.zeros(len(train_df))
    fold_scores = []

    print("\nStarting 5-Fold Stratified Cross Validation (Strict Zero-Leakage)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        df_train_f = train_df.iloc[train_idx].reset_index(drop=True)
        df_val_f = train_df.iloc[val_idx].reset_index(drop=True)

        y_train_f = y[train_idx]
        y_val_f = y[val_idx]
        y_bin_train_f = y_binary[train_idx]
        y_bin_val_f = y_binary[val_idx]

        X_tr, proc_tr, fold_cache = extract_features_v9(
            df_train_f,
            mpnet_train[train_idx],
            bge_train[train_idx],
            is_train=True,
            y_train=y_train_f
        )
        X_va, proc_va, _ = extract_features_v9(
            df_val_f,
            mpnet_train[val_idx],
            bge_train[val_idx],
            is_train=False,
            kmeans_model=fold_cache['kmeans'],
            svd_model=fold_cache['svd'],
            pos_c_mpnet=fold_cache['pos_c_mpnet'],
            top5_c_mpnet=fold_cache['top5_c_mpnet'],
            neg_c_mpnet=fold_cache['neg_c_mpnet'],
            pos_c_bge=fold_cache['pos_c_bge'],
            top5_c_bge=fold_cache['top5_c_bge'],
            neg_c_bge=fold_cache['neg_c_bge'],
            sub_c_pos=fold_cache['sub_c_pos'],
            sub_c_neg=fold_cache['sub_c_neg'],
            venue_stats=fold_cache['venue_stats'],
        )

        # Stage 1: Tri-Model Binary Classifier
        clf_lgb = lgb.LGBMClassifier(n_estimators=240, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42+fold, verbose=-1)
        clf_xgb = xgb.XGBClassifier(n_estimators=220, learning_rate=0.025, max_depth=4, random_state=42+fold, eval_metric='logloss')
        clf_lgb.fit(X_tr, y_bin_train_f)
        clf_xgb.fit(X_tr, y_bin_train_f)

        if CATBOOST_AVAILABLE:
            clf_cat = CatBoostClassifier(iterations=240, learning_rate=0.025, depth=4, random_seed=42+fold, verbose=False)
            clf_cat.fit(X_tr, y_bin_train_f)
            p_val = 0.40 * clf_lgb.predict_proba(X_va)[:, 1] + 0.40 * clf_xgb.predict_proba(X_va)[:, 1] + 0.20 * clf_cat.predict_proba(X_va)[:, 1]
        else:
            p_val = 0.50 * clf_lgb.predict_proba(X_va)[:, 1] + 0.50 * clf_xgb.predict_proba(X_va)[:, 1]

        # Stage 2: Tri-Model Continuous Regressor
        reg_lgb = lgb.LGBMRegressor(n_estimators=260, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42+fold, verbose=-1)
        reg_xgb = xgb.XGBRegressor(n_estimators=240, learning_rate=0.025, max_depth=4, random_state=42+fold)
        reg_lgb.fit(X_tr, y_train_f)
        reg_xgb.fit(X_tr, y_train_f)

        if CATBOOST_AVAILABLE:
            reg_cat = CatBoostRegressor(iterations=260, learning_rate=0.025, depth=4, random_seed=42+fold, verbose=False)
            reg_cat.fit(X_tr, y_train_f)
            s_reg_val = 0.40 * reg_lgb.predict(X_va) + 0.40 * reg_xgb.predict(X_va) + 0.20 * reg_cat.predict(X_va)
        else:
            s_reg_val = 0.50 * reg_lgb.predict(X_va) + 0.50 * reg_xgb.predict(X_va)

        # Hard zero proceedings
        p_val = np.where(proc_va == 1.0, 0.0, p_val)
        s_reg_val = np.where(proc_va == 1.0, 0.0, s_reg_val)

        oof_proba[val_idx] = p_val
        oof_reg[val_idx] = s_reg_val

        # Fold evaluation at default T=0.53
        p_fold_eval = np.where(p_val >= 0.53, s_reg_val, 0.0)
        sc, q, r = compute_competition_metric(df_val_f['id'].values, y_val_f, p_fold_eval)
        fold_scores.append(sc)
        print(f"  Fold {fold+1}/5: Score={sc:.5f} (Q={q:.5f}, R={r:.5f})")

    mean_fold_score = float(np.mean(fold_scores))
    std_fold_score = float(np.std(fold_scores))
    print(f"\nMean Fold Score: {mean_fold_score:.5f} +/- {std_fold_score:.5f}")

    # ---------------- Full Dataset Training & Test Prediction ----------------
    print("\nTraining on 100% of Training Data...")
    X_full, proc_full, full_cache = extract_features_v9(
        train_df, mpnet_train, bge_train, is_train=True, y_train=y
    )
    X_test, proc_test, _ = extract_features_v9(
        test_df, mpnet_test, bge_test, is_train=False,
        kmeans_model=full_cache['kmeans'],
        svd_model=full_cache['svd'],
        pos_c_mpnet=full_cache['pos_c_mpnet'],
        top5_c_mpnet=full_cache['top5_c_mpnet'],
        neg_c_mpnet=full_cache['neg_c_mpnet'],
        pos_c_bge=full_cache['pos_c_bge'],
        top5_c_bge=full_cache['top5_c_bge'],
        neg_c_bge=full_cache['neg_c_bge'],
        sub_c_pos=full_cache['sub_c_pos'],
        sub_c_neg=full_cache['sub_c_neg'],
        venue_stats=full_cache['venue_stats'],
    )

    # 1. Full Binary Classifier
    clf_lgb_full = lgb.LGBMClassifier(n_estimators=260, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    clf_xgb_full = xgb.XGBClassifier(n_estimators=240, learning_rate=0.025, max_depth=4, random_state=42, eval_metric='logloss')
    clf_lgb_full.fit(X_full, y_binary)
    clf_xgb_full.fit(X_full, y_binary)

    if CATBOOST_AVAILABLE:
        clf_cat_full = CatBoostClassifier(iterations=260, learning_rate=0.025, depth=4, random_seed=42, verbose=False)
        clf_cat_full.fit(X_full, y_binary)
        test_proba = 0.40 * clf_lgb_full.predict_proba(X_test)[:, 1] + 0.40 * clf_xgb_full.predict_proba(X_test)[:, 1] + 0.20 * clf_cat_full.predict_proba(X_test)[:, 1]
    else:
        test_proba = 0.50 * clf_lgb_full.predict_proba(X_test)[:, 1] + 0.50 * clf_xgb_full.predict_proba(X_test)[:, 1]

    # 2. Full Continuous Regressor
    reg_lgb_full = lgb.LGBMRegressor(n_estimators=280, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    reg_xgb_full = xgb.XGBRegressor(n_estimators=260, learning_rate=0.025, max_depth=4, random_state=42)
    reg_lgb_full.fit(X_full, y)
    reg_xgb_full.fit(X_full, y)

    if CATBOOST_AVAILABLE:
        reg_cat_full = CatBoostRegressor(iterations=280, learning_rate=0.025, depth=4, random_seed=42, verbose=False)
        reg_cat_full.fit(X_full, y)
        test_reg = 0.40 * reg_lgb_full.predict(X_test) + 0.40 * reg_xgb_full.predict(X_test) + 0.20 * reg_cat_full.predict(X_test)
    else:
        test_reg = 0.50 * reg_lgb_full.predict(X_test) + 0.50 * reg_xgb_full.predict(X_test)

    # Hard-zero proceedings
    test_proba = np.where(proc_test == 1.0, 0.0, test_proba)
    test_reg = np.where(proc_test == 1.0, 0.0, test_reg)

    # ---------------- Calibrate Threshold T* to Exact Target 381 Papers ----------------
    print("\n--- Calibrating Decision Threshold T* to Target 381 (+/- 1) Papers ---")
    threshold_candidates = np.linspace(0.40, 0.65, 251)
    best_t = 0.530
    best_diff = 9999
    best_count = 0

    for t in threshold_candidates:
        cnt = int(np.sum(test_proba >= t))
        diff = abs(cnt - 381)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)
            best_count = cnt
            if diff == 0:
                break

    # OOF score at best_t
    priorities_oof = np.where(oof_proba >= best_t, oof_reg, 0.0)
    score_oof, q_oof, r_oof = compute_competition_metric(train_df['id'].values, y, priorities_oof)

    print(f"\n=======================================================")
    print(f"🔥 V9 OOF RESULTS & CALIBRATION:")
    print(f"   Optimal Threshold T* = {best_t:.4f}")
    print(f"   OOF Overall Score    = {score_oof:.5f}")
    print(f"   Relevance (Q)        = {q_oof:.5f}")
    print(f"   Coverage (R)         = {r_oof:.5f}")
    print(f"   Mean Fold Score      = {mean_fold_score:.5f}")
    print(f"   Test Non-Zero Recs   = {best_count} / {len(test_df)} ({best_count/len(test_df)*100:.2f}%)")
    print(f"=======================================================\n")

    # Generate Final Test Priorities
    test_priorities = np.where(test_proba >= best_t, test_reg, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    sub_dir = os.path.join(BASE_DIR, "submissions")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v9_mpnet_dense.csv",
        output_dir=sub_dir
    )

    recom_count = int(np.sum(test_priorities > 0))
    zero_count = int(np.sum(test_priorities == 0))
    print(f"Saved V9 Submission: {out_file}")
    print(f"  Recommended count: {recom_count} ({recom_count/len(test_priorities)*100:.2f}%)")
    print(f"  Filtered count:    {zero_count} ({zero_count/len(test_priorities)*100:.2f}%)")
    print(f"  Priority Range:    min={test_priorities[test_priorities > 0].min():.4f}, max={test_priorities.max():.4f}, mean={test_priorities[test_priorities > 0].mean():.4f}")


if __name__ == "__main__":
    run_v9_pipeline()
