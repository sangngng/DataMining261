import os
import sys

# Ensure source-code root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from collections import defaultdict
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


def extract_features_v7(
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
    author_stats=None,
    y_train=None,
):
    """
    V7 Enhanced Dual-Embedding Pipeline:
    1. Dual-Embedding (MiniLM 384d + BGE-small-en-v1.5 384d = 768d)
    2. Multi-Centroid Cosine Similarities on both embeddings (Positive >= 4, Top-5 == 5, Negative == 1)
    3. SVD(64) latent axes + KMeans(12) topic clustering
    4. Deterministic proceedings indicator
    5. Bayesian Venue Target Encoding (m=5.0)
    6. Regularized Bayesian Author Affinity Prior (m=10.0, min_papers=3, strictly leak-free)
    """
    n_samples = len(df)

    # 1. Multi-Centroid Cosine Similarities
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

    # Blended similarities
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

    # 6. Regularized Bayesian Author Affinity Prior (m=10.0, min_papers=3)
    if is_train:
        m_author = 10.0
        author_ratings = defaultdict(list)
        for idx, r in df.iterrows():
            if pd.isna(r['authors']):
                continue
            for a in r['authors'].split(','):
                a_clean = a.strip()
                if a_clean:
                    author_ratings[a_clean].append(y_train[idx])

        author_bayes_map = {}
        for a, rats in author_ratings.items():
            if len(rats) >= 3:
                cnt = len(rats)
                mean_r = sum(rats) / cnt
                smoothed_a = (cnt * mean_r + m_author * global_mean) / (cnt + m_author)
                author_bayes_map[a] = smoothed_a

        author_stats = {
            'bayes_map': author_bayes_map,
            'global_mean': global_mean
        }

    a_b_map = author_stats['bayes_map']
    a_g_mean = author_stats['global_mean']

    def calc_author_prior(a_str):
        if pd.isna(a_str) or not str(a_str).strip():
            return a_g_mean
        scores = [a_b_map.get(a.strip(), a_g_mean) for a in str(a_str).split(',') if a.strip() in a_b_map]
        return max(scores) if scores else a_g_mean

    def calc_author_pref_flag(a_str):
        if pd.isna(a_str):
            return 0.0
        scores = [a_b_map.get(a.strip(), a_g_mean) for a in str(a_str).split(',') if a.strip() in a_b_map]
        return 1.0 if any(s >= 3.2 for s in scores) else 0.0

    author_bayes_feat = np.array([calc_author_prior(a) for a in author_series]).reshape(-1, 1)
    author_pref_flag = np.array([calc_author_pref_flag(a) for a in author_series]).reshape(-1, 1)

    # Assemble all feature columns (64 + 12 + 10 + 13 = 99 columns)
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
        author_bayes_feat,  # 1
        author_pref_flag,   # 1
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
        'author_stats': author_stats,
    }
    return X, is_proceedings.flatten(), cache


def run_v7_pipeline():
    set_seed(42)
    print("==========================================================================")
    print("🚀 LAUNCHING V7: EXPECTED GAIN CALIBRATION & ENHANCED DUAL-EMBEDDING")
    print("==========================================================================")

    data_dir = os.path.join(BASE_DIR, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Load Dual Embeddings
    mini_train = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    mini_test = np.load(os.path.join(data_dir, "test_embeddings_minilm.npy"))
    bge_train = np.load(os.path.join(data_dir, "train_embeddings_bge.npy"))
    bge_test = np.load(os.path.join(data_dir, "test_embeddings_bge.npy"))

    print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}")
    print(f"Embeddings: MiniLM={mini_train.shape}, BGE={bge_train.shape}")

    y = train_df['Label'].values
    y_binary = (y > 1).astype(int)
    y_classes = y - 1 # 0, 1, 2, 3, 4 for multiclass
    gain_weights = np.array([0.0, 0.25, 0.50, 0.75, 1.00]) # exact g_r values

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = np.zeros(len(train_df))
    oof_reg = np.zeros(len(train_df))
    oof_exp_gain = np.zeros(len(train_df))
    oof_combined_score = np.zeros(len(train_df))

    fold_scores = []
    print("\nStarting 5-Fold Stratified Cross Validation (Strict Zero-Leakage)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        df_train_f = train_df.iloc[train_idx].reset_index(drop=True)
        df_val_f = train_df.iloc[val_idx].reset_index(drop=True)

        y_train_f = y[train_idx]
        y_val_f = y[val_idx]
        y_bin_train_f = y_binary[train_idx]
        y_bin_val_f = y_binary[val_idx]
        y_cls_train_f = y_classes[train_idx]

        # Extract fold features
        X_tr, proc_tr, fold_cache = extract_features_v7(
            df_train_f,
            mini_train[train_idx],
            bge_train[train_idx],
            is_train=True,
            y_train=y_train_f
        )
        X_va, proc_va, _ = extract_features_v7(
            df_val_f,
            mini_train[val_idx],
            bge_train[val_idx],
            is_train=False,
            kmeans_model=fold_cache['kmeans'],
            svd_model=fold_cache['svd'],
            pos_centroid_mini=fold_cache['pos_centroid_mini'],
            top5_centroid_mini=fold_cache['top5_centroid_mini'],
            neg_centroid_mini=fold_cache['neg_centroid_mini'],
            pos_centroid_bge=fold_cache['pos_centroid_bge'],
            top5_centroid_bge=fold_cache['top5_centroid_bge'],
            neg_centroid_bge=fold_cache['neg_centroid_bge'],
            venue_stats=fold_cache['venue_stats'],
            author_stats=fold_cache['author_stats'],
        )

        # Stage 1: Tri-Model Binary Classification
        clf_lgb = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42+fold, verbose=-1)
        clf_xgb = xgb.XGBClassifier(n_estimators=200, learning_rate=0.025, max_depth=4, random_state=42+fold, eval_metric='logloss')
        clf_lgb.fit(X_tr, y_bin_train_f)
        clf_xgb.fit(X_tr, y_bin_train_f)

        if CATBOOST_AVAILABLE:
            clf_cat = CatBoostClassifier(iterations=220, learning_rate=0.025, depth=4, random_seed=42+fold, verbose=False)
            clf_cat.fit(X_tr, y_bin_train_f)
            p_val = 0.40 * clf_lgb.predict_proba(X_va)[:, 1] + 0.40 * clf_xgb.predict_proba(X_va)[:, 1] + 0.20 * clf_cat.predict_proba(X_va)[:, 1]
        else:
            p_val = 0.50 * clf_lgb.predict_proba(X_va)[:, 1] + 0.50 * clf_xgb.predict_proba(X_va)[:, 1]

        # Stage 2A: Continuous Regressors
        reg_lgb = lgb.LGBMRegressor(n_estimators=240, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42+fold, verbose=-1)
        reg_xgb = xgb.XGBRegressor(n_estimators=220, learning_rate=0.025, max_depth=4, random_state=42+fold)
        reg_lgb.fit(X_tr, y_train_f)
        reg_xgb.fit(X_tr, y_train_f)

        if CATBOOST_AVAILABLE:
            reg_cat = CatBoostRegressor(iterations=240, learning_rate=0.025, depth=4, random_seed=42+fold, verbose=False)
            reg_cat.fit(X_tr, y_train_f)
            s_reg_val = 0.40 * reg_lgb.predict(X_va) + 0.40 * reg_xgb.predict(X_va) + 0.20 * reg_cat.predict(X_va)
        else:
            s_reg_val = 0.50 * reg_lgb.predict(X_va) + 0.50 * reg_xgb.predict(X_va)

        # Stage 2B: Multiclass Expected Gain Estimation
        clf_multi_lgb = lgb.LGBMClassifier(objective='multiclass', num_class=5, n_estimators=200, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42+fold, verbose=-1)
        clf_multi_lgb.fit(X_tr, y_cls_train_f)
        probs_multi_val = clf_multi_lgb.predict_proba(X_va)

        if CATBOOST_AVAILABLE:
            clf_multi_cat = CatBoostClassifier(iterations=200, learning_rate=0.025, depth=4, loss_function='MultiClass', random_seed=42+fold, verbose=False)
            clf_multi_cat.fit(X_tr, y_cls_train_f)
            probs_multi_val = 0.60 * probs_multi_val + 0.40 * clf_multi_cat.predict_proba(X_va)

        exp_gain_val = np.dot(probs_multi_val, gain_weights)
        s_exp_gain_val = 1.0 + 4.0 * exp_gain_val # mapped to continuous 1.0-5.0 priority scale

        # Fused Priority Score (50% Continuous Regressor + 50% Expected Gain)
        s_fused_val = 0.50 * s_reg_val + 0.50 * s_exp_gain_val

        # Hard-zero proceedings
        p_val = np.where(proc_va == 1.0, 0.0, p_val)
        s_fused_val = np.where(proc_va == 1.0, 0.0, s_fused_val)

        oof_proba[val_idx] = p_val
        oof_reg[val_idx] = s_reg_val
        oof_exp_gain[val_idx] = s_exp_gain_val
        oof_combined_score[val_idx] = s_fused_val

        # Fold evaluation at default T=0.53
        p_fold_eval = np.where(p_val >= 0.53, s_fused_val, 0.0)
        sc, q, r = compute_competition_metric(df_val_f['id'].values, y_val_f, p_fold_eval)
        fold_scores.append(sc)
        print(f"  Fold {fold+1}/5: Score={sc:.5f} (Q={q:.5f}, R={r:.5f})")

    mean_fold_score = float(np.mean(fold_scores))
    std_fold_score = float(np.std(fold_scores))
    print(f"\nMean Fold Score: {mean_fold_score:.5f} +/- {std_fold_score:.5f}")

    # ---------------- Full Dataset Training & Test Prediction ----------------
    print("\nTraining on 100% of Training Data...")
    X_full, proc_full, full_cache = extract_features_v7(
        train_df, mini_train, bge_train, is_train=True, y_train=y
    )
    X_test, proc_test, _ = extract_features_v7(
        test_df, mini_test, bge_test, is_train=False,
        kmeans_model=full_cache['kmeans'],
        svd_model=full_cache['svd'],
        pos_centroid_mini=full_cache['pos_centroid_mini'],
        top5_centroid_mini=full_cache['top5_centroid_mini'],
        neg_centroid_mini=full_cache['neg_centroid_mini'],
        pos_centroid_bge=full_cache['pos_centroid_bge'],
        top5_centroid_bge=full_cache['top5_centroid_bge'],
        neg_centroid_bge=full_cache['neg_centroid_bge'],
        venue_stats=full_cache['venue_stats'],
        author_stats=full_cache['author_stats'],
    )

    # 1. Full Binary Classifier
    clf_lgb_full = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    clf_xgb_full = xgb.XGBClassifier(n_estimators=220, learning_rate=0.025, max_depth=4, random_state=42, eval_metric='logloss')
    clf_lgb_full.fit(X_full, y_binary)
    clf_xgb_full.fit(X_full, y_binary)

    if CATBOOST_AVAILABLE:
        clf_cat_full = CatBoostClassifier(iterations=250, learning_rate=0.025, depth=4, random_seed=42, verbose=False)
        clf_cat_full.fit(X_full, y_binary)
        test_proba = 0.40 * clf_lgb_full.predict_proba(X_test)[:, 1] + 0.40 * clf_xgb_full.predict_proba(X_test)[:, 1] + 0.20 * clf_cat_full.predict_proba(X_test)[:, 1]
    else:
        test_proba = 0.50 * clf_lgb_full.predict_proba(X_test)[:, 1] + 0.50 * clf_xgb_full.predict_proba(X_test)[:, 1]

    # 2. Full Continuous Regressor
    reg_lgb_full = lgb.LGBMRegressor(n_estimators=260, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    reg_xgb_full = xgb.XGBRegressor(n_estimators=240, learning_rate=0.025, max_depth=4, random_state=42)
    reg_lgb_full.fit(X_full, y)
    reg_xgb_full.fit(X_full, y)

    if CATBOOST_AVAILABLE:
        reg_cat_full = CatBoostRegressor(iterations=260, learning_rate=0.025, depth=4, random_seed=42, verbose=False)
        reg_cat_full.fit(X_full, y)
        test_reg = 0.40 * reg_lgb_full.predict(X_test) + 0.40 * reg_xgb_full.predict(X_test) + 0.20 * reg_cat_full.predict(X_test)
    else:
        test_reg = 0.50 * reg_lgb_full.predict(X_test) + 0.50 * reg_xgb_full.predict(X_test)

    # 3. Full Multiclass Expected Gain
    clf_multi_lgb_full = lgb.LGBMClassifier(objective='multiclass', num_class=5, n_estimators=220, learning_rate=0.025, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    clf_multi_lgb_full.fit(X_full, y_classes)
    test_probs_multi = clf_multi_lgb_full.predict_proba(X_test)

    if CATBOOST_AVAILABLE:
        clf_multi_cat_full = CatBoostClassifier(iterations=220, learning_rate=0.025, depth=4, loss_function='MultiClass', random_seed=42, verbose=False)
        clf_multi_cat_full.fit(X_full, y_classes)
        test_probs_multi = 0.60 * test_probs_multi + 0.40 * clf_multi_cat_full.predict_proba(X_test)

    test_exp_gain = np.dot(test_probs_multi, gain_weights)
    test_s_exp_gain = 1.0 + 4.0 * test_exp_gain

    test_fused_score = 0.50 * test_reg + 0.50 * test_s_exp_gain

    # Hard-zero proceedings in test
    test_proba = np.where(proc_test == 1.0, 0.0, test_proba)
    test_fused_score = np.where(proc_test == 1.0, 0.0, test_fused_score)

    # ---------------- Calibrate Threshold T* in Golden Zone [370, 382] ----------------
    print("\n--- Sweeping Threshold T* Constrained to Golden Zone [370, 382] ---")
    best_score = -1.0
    best_t = 0.530
    best_q = 0.0
    best_r = 0.0
    best_test_count = 0

    threshold_candidates = np.linspace(0.40, 0.65, 101)
    golden_min, golden_max = 370, 382

    for t in threshold_candidates:
        test_count = int(np.sum(test_proba >= t))
        if golden_min <= test_count <= golden_max:
            priorities_oof = np.where(oof_proba >= t, oof_combined_score, 0.0)
            score, q, r = compute_competition_metric(train_df['id'].values, y, priorities_oof)
            if score > best_score:
                best_score = score
                best_t = float(t)
                best_q = q
                best_r = r
                best_test_count = test_count

    print(f"\n=======================================================")
    print(f"🔥 V7 OOF RESULTS & CALIBRATION:")
    print(f"   Optimal Threshold T* = {best_t:.4f}")
    print(f"   OOF Overall Score    = {best_score:.5f}")
    print(f"   Relevance (Q)        = {best_q:.5f}")
    print(f"   Coverage (R)         = {best_r:.5f}")
    print(f"   Mean Fold Score      = {mean_fold_score:.5f}")
    print(f"   Test Non-Zero Recs   = {best_test_count} / {len(test_df)} ({best_test_count/len(test_df)*100:.2f}%)")
    print(f"=======================================================\n")

    # Generate Final Test Priorities
    test_priorities = np.where(test_proba >= best_t, test_fused_score, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    sub_dir = os.path.join(BASE_DIR, "submissions")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v7_expected_gain.csv",
        output_dir=sub_dir
    )

    recom_count = int(np.sum(test_priorities > 0))
    zero_count = int(np.sum(test_priorities == 0))
    print(f"Saved V7 Submission: {out_file}")
    print(f"  Recommended count: {recom_count} ({recom_count/len(test_priorities)*100:.2f}%)")
    print(f"  Filtered count:    {zero_count} ({zero_count/len(test_priorities)*100:.2f}%)")
    print(f"  Priority Range:    min={test_priorities[test_priorities > 0].min():.4f}, max={test_priorities.max():.4f}, mean={test_priorities[test_priorities > 0].mean():.4f}")


if __name__ == "__main__":
    run_v7_pipeline()
