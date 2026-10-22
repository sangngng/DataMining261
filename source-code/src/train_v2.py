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
from typing import Tuple

from src.metric import compute_competition_metric
from src.utils import set_seed, save_submission


def extract_features(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    is_train: bool = True,
    kmeans_model = None,
    svd_model = None,
    pos_centroid = None,
    neg_centroid = None,
    venue_stats = None,
    author_stats = None,
    y_train = None
):
    """
    Extract enhanced semantic and metadata features strictly avoiding data leakage.
    """
    n_samples = len(df)
    
    # 1. Cosine similarity to Centroids
    if is_train:
        pos_mask = (y_train >= 4)
        neg_mask = (y_train == 1)
        pos_centroid = embeddings[pos_mask].mean(axis=0, keepdims=True)
        neg_centroid = embeddings[neg_mask].mean(axis=0, keepdims=True)
        # normalize
        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-9)
        neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-9)

    sim_pos = np.dot(embeddings, pos_centroid.T) # shape (N, 1)
    sim_neg = np.dot(embeddings, neg_centroid.T) # shape (N, 1)
    sim_diff = sim_pos - sim_neg

    # 2. SVD Dimensionality Reduction on SBERT (capture top 48 latent semantic axes)
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

    # 4. Metadata features
    # Year
    year_num = pd.to_numeric(df['year'], errors='coerce').fillna(2020.0).values.reshape(-1, 1)
    year_norm = (year_num - 2000.0) / 26.0
    paper_age = (2026.0 - year_num) / 10.0
    is_recent = (year_num >= 2024.0).astype(float)

    # DOI
    has_doi = (~df['doi'].isna()).astype(float).values.reshape(-1, 1)

    # Authors
    author_series = df['authors'].fillna('')
    author_count = author_series.apply(lambda s: len(s.split(',')) if len(s.strip()) > 0 else 0).values.reshape(-1, 1)
    is_single_author = (author_count == 1).astype(float)

    # Title length
    title_series = df['title'].fillna('')
    char_len = (title_series.apply(len).values.reshape(-1, 1) - 50.0) / 50.0
    word_count = (title_series.apply(lambda s: len(s.split())).values.reshape(-1, 1) - 10.0) / 10.0

    # Venue Target Encoding
    venue_series = df['venue'].fillna('UNKNOWN').astype(str)
    if is_train:
        venue_mean = df.groupby('venue')['Label'].mean().to_dict()
        venue_counts = venue_series.value_counts().to_dict()
        venue_stats = {'mean': venue_mean, 'counts': venue_counts, 'global_mean': float(y_train.mean())}
    
    v_mean_map = venue_stats['mean']
    v_cnt_map = venue_stats['counts']
    g_mean = venue_stats['global_mean']

    v_enc = venue_series.map(v_mean_map).fillna(g_mean).values.reshape(-1, 1)
    v_freq = np.log1p(venue_series.map(v_cnt_map).fillna(0.0).values.reshape(-1, 1))

    # Assemble all features
    X = np.hstack([
        svd_feats,           # 48
        cluster_dists,       # 12
        sim_pos,             # 1
        sim_neg,             # 1
        sim_diff,            # 1
        year_norm,           # 1
        paper_age,           # 1
        is_recent,           # 1
        has_doi,             # 1
        author_count,        # 1
        is_single_author,    # 1
        char_len,            # 1
        word_count,          # 1
        v_enc,               # 1
        v_freq               # 1
    ])

    cache = {
        'kmeans': kmeans_model,
        'svd': svd_model,
        'pos_centroid': pos_centroid,
        'neg_centroid': neg_centroid,
        'venue_stats': venue_stats
    }
    return X, cache

def run_v2_pipeline():
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
    
    oof_proba_lgb = np.zeros(len(train_df))
    oof_proba_xgb = np.zeros(len(train_df))
    oof_reg_lgb = np.zeros(len(train_df))
    oof_reg_xgb = np.zeros(len(train_df))
    
    print("\n========== RUNNING 5-FOLD CV: ENHANCED PIPELINE (SBERT + GBDT ENSEMBLE) ==========")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y), 1):
        df_tr, df_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        yb_tr, yb_val = y_binary[train_idx], y_binary[val_idx]
        
        emb_tr, emb_val = train_emb[train_idx], train_emb[val_idx]
        
        # Extract features fitted strictly on train fold
        X_tr, cache = extract_features(df_tr, emb_tr, is_train=True, y_train=y_tr)
        X_val, _ = extract_features(
            df_val, emb_val, is_train=False,
            kmeans_model=cache['kmeans'],
            svd_model=cache['svd'],
            pos_centroid=cache['pos_centroid'],
            neg_centroid=cache['neg_centroid'],
            venue_stats=cache['venue_stats']
        )
        
        # 1. Models - LightGBM
        clf_lgb = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
        reg_lgb = lgb.LGBMRegressor(n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
        
        clf_lgb.fit(X_tr, yb_tr)
        reg_lgb.fit(X_tr, y_tr)
        
        # 2. Models - XGBoost
        clf_xgb = xgb.XGBClassifier(n_estimators=180, learning_rate=0.03, max_depth=4, random_state=42, eval_metric='logloss')
        reg_xgb = xgb.XGBRegressor(n_estimators=180, learning_rate=0.03, max_depth=4, random_state=42)
        
        clf_xgb.fit(X_tr, yb_tr)
        reg_xgb.fit(X_tr, y_tr)
        
        # OOF Predictions
        oof_proba_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]
        oof_proba_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
        oof_reg_lgb[val_idx] = reg_lgb.predict(X_val)
        oof_reg_xgb[val_idx] = reg_xgb.predict(X_val)
        
        # Fold evaluation (Ensemble 50-50)
        p_fold_proba = 0.5 * oof_proba_lgb[val_idx] + 0.5 * oof_proba_xgb[val_idx]
        p_fold_reg = 0.5 * oof_reg_lgb[val_idx] + 0.5 * oof_reg_xgb[val_idx]
        p_fold = np.where(p_fold_proba >= 0.35, p_fold_reg, 0.0)
        
        sc, q, r = compute_competition_metric(df_val['id'].values, y_val, p_fold)
        print(f"Fold {fold}: Score = {sc:.5f} (Q = {q:.5f}, R = {r:.5f})")

    # Overall OOF Evaluation & Threshold Sweep
    oof_proba_ens = 0.5 * oof_proba_lgb + 0.5 * oof_proba_xgb
    oof_reg_ens = 0.5 * oof_reg_lgb + 0.5 * oof_reg_xgb
    
    best_t = 0.35
    best_score = 0.0
    best_q = 0.0
    best_r = 0.0
    
    print("\n--- Sweeping Threshold T on Out-Of-Fold Predictions ---")
    for t in np.linspace(0.15, 0.80, 27):
        p_oof = np.where(oof_proba_ens >= t, oof_reg_ens, 0.0)
        p_oof = np.clip(p_oof, 0.0, None)
        sc, q, r = compute_competition_metric(train_df['id'].values, y, p_oof)
        if sc > best_score:
            best_score = sc
            best_q = q
            best_r = r
            best_t = t

    print(f"\n=======================================================")
    print(f"🔥 V2 PIPELINE OOF RESULTS:")
    print(f"   Optimal Threshold T* = {best_t:.2f}")
    print(f"   OOF Overall Score    = {best_score:.5f}")
    print(f"   Relevance (Q)        = {best_q:.5f}")
    print(f"   Coverage (R)         = {best_r:.5f}")
    print(f"   (Baseline score was:   0.80370)")
    print(f"   Delta vs Baseline    : {best_score - 0.80370:+.5f}")
    print(f"=======================================================\n")

    # Train on Full Dataset and Predict on Test Set
    print("Training on Full Dataset with Optimal T*...")
    X_full, full_cache = extract_features(train_df, train_emb, is_train=True, y_train=y)
    X_test, _ = extract_features(
        test_df, test_emb, is_train=False,
        kmeans_model=full_cache['kmeans'],
        svd_model=full_cache['svd'],
        pos_centroid=full_cache['pos_centroid'],
        neg_centroid=full_cache['neg_centroid'],
        venue_stats=full_cache['venue_stats']
    )
    
    # Fit full models
    clf_lgb_full = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    reg_lgb_full = lgb.LGBMRegressor(n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=24, random_state=42, verbose=-1)
    clf_lgb_full.fit(X_full, y_binary)
    reg_lgb_full.fit(X_full, y)
    
    clf_xgb_full = xgb.XGBClassifier(n_estimators=180, learning_rate=0.03, max_depth=4, random_state=42, eval_metric='logloss')
    reg_xgb_full = xgb.XGBRegressor(n_estimators=180, learning_rate=0.03, max_depth=4, random_state=42)
    clf_xgb_full.fit(X_full, y_binary)
    reg_xgb_full.fit(X_full, y)
    
    test_proba = 0.5 * clf_lgb_full.predict_proba(X_test)[:, 1] + 0.5 * clf_xgb_full.predict_proba(X_test)[:, 1]
    test_reg = 0.5 * reg_lgb_full.predict(X_test) + 0.5 * reg_xgb_full.predict(X_test)
    
    test_priorities = np.where(test_proba >= best_t, test_reg, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)
    
    # Save submission v2
    sub_dir = os.path.join(base_dir, "submissions")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v2_sbert_ensemble.csv",
        output_dir=sub_dir
    )
    
    recom_count = np.sum(test_priorities > 0)
    print(f"\nRecommended test papers (priority > 0): {recom_count} / {len(test_priorities)}")
    print(f"Filtered out test papers (priority == 0): {len(test_priorities) - recom_count} / {len(test_priorities)}")
    print(f"\n[DONE] Saved new submission to: {out_file}")

if __name__ == "__main__":
    run_v2_pipeline()
