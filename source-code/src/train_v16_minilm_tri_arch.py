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
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.metric import compute_competition_metric
from src.utils import set_seed, save_submission

# =============================================================================
# V16 MINI-LM CENTRIC TRI-ARCHITECTURE PIPELINE
#
# Scientific Foundation & Lessons Applied:
#   1. Pure MiniLM Embedding Space (Zero Semantic Drift):
#      All empirical evidence from V2, V4, V8, V9, V15 demonstrates that
#      all-MiniLM-L6-v2 (384d) is the optimal representation space for this
#      domain. BGE and MPNet cause semantic drift on theoretical LICS/CAV papers.
#      Therefore, V16 anchors 100% on MiniLM.
#   2. Architectural Diversity without Cross-Embedding Noise:
#      - Pipeline 1 (Global Topic & Centroids): SVD(48) + KMeans(12) +
#        3-way Centroids + LightGBM & CatBoost.
#      - Pipeline 2 (Multi-Modal Sub-Centroids): SVD(64) + 8 Pos Sub-Centroids
#        (KMeans on Label>=3) + Margin + XGBoost & CatBoost.
#      - Pipeline 3 (Manifold Density & Nearest Neighbors): SVD(48) +
#        k-NN local label density (k=5, 15) + distance to nearest 5-star vs 1-star
#        + Bayesian Venue prior + LightGBM & XGBoost.
#   3. Strict 10-Fold Stratified Cross-Validation (Zero Data Leakage):
#      All transformers, clusters, centroids, and k-NN indices are fit strictly
#      on the fold-train split.
#   4. Metric-Driven Late Fusion:
#      Ensemble weights (alpha, beta, gamma) optimized via OOF grid search
#      directly maximizing compute_competition_metric.
#   5. Golden Zone Shortlist Calibration:
#      Test threshold T* calibrated to strictly select 381 papers (63.93%).
# =============================================================================

N_SPLITS = 10
SEED = 42
TARGET_TEST_COUNT = 381

PROCEEDINGS_KEYWORDS = [
    'proceedings', 'conference on', 'preface', 'workshop on',
    'doctoral consortium', 'front matter', 'editorial'
]


# -----------------------------------------------------------------------------
# Clean Structural Metadata
# -----------------------------------------------------------------------------
def extract_clean_metadata(df: pd.DataFrame) -> np.ndarray:
    year_num = pd.to_numeric(df['year'], errors='coerce').fillna(2020.0).values.reshape(-1, 1)
    year_norm = (year_num - 2000.0) / 26.0
    paper_age = (2026.0 - year_num) / 10.0
    is_recent = (year_num >= 2024.0).astype(float)

    has_doi = (~df['doi'].isna()).astype(float).values.reshape(-1, 1)

    author_series = df['authors'].fillna('')
    author_count = author_series.apply(
        lambda s: len(s.split(',')) if len(s.strip()) > 0 else 0
    ).values.reshape(-1, 1)
    is_single_author = (author_count == 1).astype(float)

    title_series = df['title'].fillna('')
    char_len = (title_series.apply(len).values.reshape(-1, 1) - 50.0) / 50.0
    word_count = (title_series.apply(lambda s: len(s.split())).values.reshape(-1, 1) - 10.0) / 10.0

    is_proceedings = title_series.str.lower().apply(
        lambda s: float(any(kw in s for kw in PROCEEDINGS_KEYWORDS))
    ).values.reshape(-1, 1)

    return np.hstack([
        year_norm, paper_age, is_recent, has_doi,
        author_count, is_single_author, char_len, word_count, is_proceedings
    ])


# -----------------------------------------------------------------------------
# Pipeline 1: Global Topic & Centroids (V4 Clean Core)
# Models: LightGBM + CatBoost
# -----------------------------------------------------------------------------
def extract_features_p1(df, emb, meta, is_train, y_train=None, cache=None):
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        pos_mask = (y_train >= 4)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        pos_c = emb[pos_mask].mean(axis=0, keepdims=True)
        top5_c = emb[top5_mask].mean(axis=0, keepdims=True)
        neg_c = emb[neg_mask].mean(axis=0, keepdims=True)
        pos_c = pos_c / (np.linalg.norm(pos_c) + 1e-9)
        top5_c = top5_c / (np.linalg.norm(top5_c) + 1e-9)
        neg_c = neg_c / (np.linalg.norm(neg_c) + 1e-9)

        svd_model = TruncatedSVD(n_components=48, random_state=SEED)
        svd_feats = svd_model.fit_transform(emb)

        kmeans_model = KMeans(n_clusters=12, random_state=SEED, n_init=10)
        cluster_dists = kmeans_model.fit_transform(emb)

        cache = {'svd': svd_model, 'kmeans': kmeans_model,
                 'pos_c': pos_c, 'top5_c': top5_c, 'neg_c': neg_c}
    else:
        svd_feats = cache['svd'].transform(emb)
        cluster_dists = cache['kmeans'].transform(emb)
        pos_c, top5_c, neg_c = cache['pos_c'], cache['top5_c'], cache['neg_c']

    sim_pos = np.dot(emb, pos_c.T)
    sim_top5 = np.dot(emb, top5_c.T)
    sim_neg = np.dot(emb, neg_c.T)
    sim_diff = sim_pos - sim_neg

    is_recent = meta[:, 2:3]
    has_doi = meta[:, 3:4]
    sim_recent_inter = sim_diff * is_recent
    sim_doi_inter = sim_diff * has_doi

    X = np.hstack([svd_feats, cluster_dists, sim_pos, sim_top5, sim_neg, sim_diff,
                   sim_recent_inter, sim_doi_inter, meta])
    return X, cache


# -----------------------------------------------------------------------------
# Pipeline 2: Multi-Modal Sub-Centroids on Positives (K=8)
# Models: XGBoost + CatBoost
# -----------------------------------------------------------------------------
def extract_features_p2(df, emb, meta, is_train, y_train=None, cache=None):
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        pos_mask = (y_train >= 3)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        neg_c = emb[neg_mask].mean(axis=0, keepdims=True)
        neg_c = neg_c / (np.linalg.norm(neg_c) + 1e-9)

        top5_c = emb[top5_mask].mean(axis=0, keepdims=True)
        top5_c = top5_c / (np.linalg.norm(top5_c) + 1e-9)

        km_sub = KMeans(n_clusters=8, random_state=SEED, n_init=10)
        km_sub.fit(emb[pos_mask])
        sub_centroids = km_sub.cluster_centers_
        sub_centroids = sub_centroids / (np.linalg.norm(sub_centroids, axis=1, keepdims=True) + 1e-9)

        svd_model = TruncatedSVD(n_components=64, random_state=SEED)
        svd_feats = svd_model.fit_transform(emb)

        cache = {'svd': svd_model, 'sub_centroids': sub_centroids,
                 'top5_c': top5_c, 'neg_c': neg_c}
    else:
        svd_feats = cache['svd'].transform(emb)
        sub_centroids = cache['sub_centroids']
        top5_c, neg_c = cache['top5_c'], cache['neg_c']

    sub_sims = np.dot(emb, sub_centroids.T)  # (N, 8)
    max_sub_sim = sub_sims.max(axis=1, keepdims=True)
    mean_sub_sim = sub_sims.mean(axis=1, keepdims=True)
    sim_top5 = np.dot(emb, top5_c.T)
    sim_neg = np.dot(emb, neg_c.T)
    margin_sub = max_sub_sim - sim_neg
    margin_top5 = sim_top5 - sim_neg

    X = np.hstack([svd_feats, sub_sims, max_sub_sim, mean_sub_sim, sim_top5, sim_neg,
                   margin_sub, margin_top5, meta])
    return X, cache


# -----------------------------------------------------------------------------
# Pipeline 3: Manifold Density & Nearest Neighbors Graph
# Models: LightGBM + XGBoost
# -----------------------------------------------------------------------------
def extract_features_p3(df, emb, meta, is_train, y_train=None, cache=None, venue_series=None):
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        
        # 1. SVD 48
        svd_model = TruncatedSVD(n_components=48, random_state=SEED)
        svd_feats = svd_model.fit_transform(emb)

        # 2. k-NN Index on Train Embeddings
        nn_model = NearestNeighbors(n_neighbors=21, metric='cosine', algorithm='brute')
        nn_model.fit(emb)

        # 3. Nearest Positive and Nearest Negative Exemplars
        pos_exemplars = emb[y_train >= 4]
        neg_exemplars = emb[y_train == 1]
        nn_pos = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute').fit(pos_exemplars)
        nn_neg = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute').fit(neg_exemplars)

        # 4. Bayesian Venue Encoding (m=5 smoothing)
        global_mean = float(np.mean(y_train > 1.0))
        venue_df = pd.DataFrame({'venue': df['venue'], 'is_pos': (y_train > 1.0).astype(float)})
        venue_stats = venue_df.groupby('venue')['is_pos'].agg(['count', 'mean'])
        m = 5.0
        venue_bayes = (venue_stats['count'] * venue_stats['mean'] + m * global_mean) / (venue_stats['count'] + m)
        venue_map = venue_bayes.to_dict()

        cache = {
            'svd': svd_model,
            'nn': nn_model,
            'nn_pos': nn_pos,
            'nn_neg': nn_neg,
            'y_train': y_train,
            'venue_map': venue_map,
            'global_mean': global_mean,
            'emb_train': emb
        }
        
        # For training fold: query neighbors excluding self
        dists, indices = nn_model.kneighbors(emb, n_neighbors=21)
        indices = indices[:, 1:] # exclude self
        dists = dists[:, 1:]
    else:
        svd_feats = cache['svd'].transform(emb)
        dists, indices = cache['nn'].kneighbors(emb, n_neighbors=20)
        y_train = cache['y_train']

    # Local label density features from nearest neighbors
    nbr_labels = y_train[indices] # (N, 20)
    pos_ratio_5 = (nbr_labels[:, :5] >= 2).mean(axis=1, keepdims=True)
    pos_ratio_15 = (nbr_labels[:, :15] >= 2).mean(axis=1, keepdims=True)
    top_ratio_5 = (nbr_labels[:, :5] >= 4).mean(axis=1, keepdims=True)
    top_ratio_15 = (nbr_labels[:, :15] >= 4).mean(axis=1, keepdims=True)
    neg_ratio_5 = (nbr_labels[:, :5] == 1).mean(axis=1, keepdims=True)
    local_mean_5 = (nbr_labels[:, :5].mean(axis=1, keepdims=True) - 1.0) / 4.0
    local_mean_15 = (nbr_labels[:, :15].mean(axis=1, keepdims=True) - 1.0) / 4.0
    mean_dist_5 = dists[:, :5].mean(axis=1, keepdims=True)

    # Cosine distance to nearest 5-star vs nearest 1-star exemplar
    d_pos, _ = cache['nn_pos'].kneighbors(emb)
    d_neg, _ = cache['nn_neg'].kneighbors(emb)
    margin_exemplar = d_neg - d_pos

    # Venue Bayesian feature
    venue_feat = df['venue'].map(cache['venue_map']).fillna(cache['global_mean']).values.reshape(-1, 1)

    X = np.hstack([
        svd_feats, pos_ratio_5, pos_ratio_15, top_ratio_5, top_ratio_15,
        neg_ratio_5, local_mean_5, local_mean_15, mean_dist_5,
        d_pos, d_neg, margin_exemplar, venue_feat, meta
    ])
    return X, cache


# -----------------------------------------------------------------------------
# Dual-Head Model Builders
# -----------------------------------------------------------------------------
def build_dual_head_p1(seed: int):
    """LightGBM + CatBoost"""
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=240, learning_rate=0.03, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=seed, verbose=-1
    )
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=240, learning_rate=0.03, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=seed, verbose=-1
    )
    heads = {'clf': {'lgb': clf_lgb}, 'reg': {'lgb': reg_lgb}}
    if CATBOOST_AVAILABLE:
        heads['clf']['cat'] = CatBoostClassifier(
            iterations=240, learning_rate=0.03, depth=5,
            subsample=0.85, colsample_bylevel=0.75, bootstrap_type='Bernoulli',
            random_seed=seed, verbose=False
        )
        heads['reg']['cat'] = CatBoostRegressor(
            iterations=240, learning_rate=0.03, depth=5,
            subsample=0.85, colsample_bylevel=0.75, bootstrap_type='Bernoulli',
            random_seed=seed, verbose=False
        )
    return heads


def build_dual_head_p2(seed: int):
    """XGBoost + CatBoost"""
    clf_xgb = xgb.XGBClassifier(
        n_estimators=220, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.75, random_state=seed,
        eval_metric='logloss', verbosity=0
    )
    reg_xgb = xgb.XGBRegressor(
        n_estimators=220, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.75, random_state=seed,
        verbosity=0
    )
    heads = {'clf': {'xgb': clf_xgb}, 'reg': {'xgb': reg_xgb}}
    if CATBOOST_AVAILABLE:
        heads['clf']['cat'] = CatBoostClassifier(
            iterations=220, learning_rate=0.03, depth=5,
            subsample=0.85, colsample_bylevel=0.75, bootstrap_type='Bernoulli',
            random_seed=seed, verbose=False
        )
        heads['reg']['cat'] = CatBoostRegressor(
            iterations=220, learning_rate=0.03, depth=5,
            subsample=0.85, colsample_bylevel=0.75, bootstrap_type='Bernoulli',
            random_seed=seed, verbose=False
        )
    return heads


def build_dual_head_p3(seed: int):
    """LightGBM + XGBoost with Regularized Settings"""
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=18,
        min_child_samples=30, subsample=0.80, colsample_bytree=0.70,
        random_state=seed, verbose=-1
    )
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=18,
        min_child_samples=30, subsample=0.80, colsample_bytree=0.70,
        random_state=seed, verbose=-1
    )
    clf_xgb = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=4,
        subsample=0.80, colsample_bytree=0.70, random_state=seed,
        eval_metric='logloss', verbosity=0
    )
    reg_xgb = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=4,
        subsample=0.80, colsample_bytree=0.70, random_state=seed,
        verbosity=0
    )
    return {
        'clf': {'lgb': clf_lgb, 'xgb': clf_xgb},
        'reg': {'lgb': reg_lgb, 'xgb': reg_xgb}
    }


def fit_dual_head(heads, X_tr, is_pos_tr, y_tr):
    for model in heads['clf'].values():
        model.fit(X_tr, is_pos_tr)
    for model in heads['reg'].values():
        model.fit(X_tr, y_tr)
    return heads


def predict_dual_head(heads, X):
    clf_probs = [model.predict_proba(X)[:, 1] for model in heads['clf'].values()]
    prob = np.mean(clf_probs, axis=0)

    reg_preds = [model.predict(X) for model in heads['reg'].values()]
    reg = np.mean(reg_preds, axis=0)
    return prob, reg


# -----------------------------------------------------------------------------
# Pipeline Specifications
# -----------------------------------------------------------------------------
PIPELINE_SPECS = {
    'p1_topic': {
        'extractor': extract_features_p1,
        'head_builder': build_dual_head_p1,
        'desc': 'Global Topic & Centroids (LightGBM + CatBoost)'
    },
    'p2_subcluster': {
        'extractor': extract_features_p2,
        'head_builder': build_dual_head_p2,
        'desc': 'Positive Sub-Centroids K=8 (XGBoost + CatBoost)'
    },
    'p3_density': {
        'extractor': extract_features_p3,
        'head_builder': build_dual_head_p3,
        'desc': 'Manifold k-NN Density & Margin (LightGBM + XGBoost)'
    },
}


# -----------------------------------------------------------------------------
# 10-Fold Stratified CV Execution
# -----------------------------------------------------------------------------
def run_pipeline_cv(name, spec, train_df, test_df, emb_train, emb_test, y, is_pos, meta_train, meta_test):
    print(f"\n{'='*75}\n🧩 Pipeline [{name.upper()}]: {spec['desc']}\n{'='*75}")
    N = len(train_df)
    oof_prob = np.zeros(N)
    oof_reg = np.zeros(N)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_scores = []

    test_prob_folds = []
    test_reg_folds = []

    extractor = spec['extractor']
    head_builder = spec['head_builder']

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
        y_tr, y_val = y[tr_idx], y[val_idx]
        is_pos_tr = is_pos[tr_idx]

        X_tr, cache = extractor(
            train_df.iloc[tr_idx], emb_train[tr_idx], meta_train[tr_idx],
            is_train=True, y_train=y_tr
        )
        X_val, _ = extractor(
            train_df.iloc[val_idx], emb_train[val_idx], meta_train[val_idx],
            is_train=False, cache=cache
        )
        X_test_fold, _ = extractor(
            test_df, emb_test, meta_test,
            is_train=False, cache=cache
        )

        heads = head_builder(seed=SEED + fold)
        fit_dual_head(heads, X_tr, is_pos_tr, y_tr)
        prob_val, reg_val = predict_dual_head(heads, X_val)
        prob_test, reg_test = predict_dual_head(heads, X_test_fold)

        oof_prob[val_idx] = prob_val
        oof_reg[val_idx] = reg_val
        test_prob_folds.append(prob_test)
        test_reg_folds.append(reg_test)

        fold_priority = np.where(prob_val >= 0.5, reg_val, 0.0)
        fold_priority = np.clip(fold_priority, 0.0, None)
        sc, q, r = compute_competition_metric(train_df['id'].values[val_idx], y_val, fold_priority)
        fold_scores.append(sc)
        print(f"  Fold {fold+1:2d}/{N_SPLITS}: Score={sc:.5f} (Q={q:.5f}, R={r:.5f})")

    test_prob = np.mean(test_prob_folds, axis=0)
    test_reg = np.mean(test_reg_folds, axis=0)

    print(f"  --> Mean Fold Score: {np.mean(fold_scores):.5f} (std={np.std(fold_scores):.5f})")

    return {
        'oof_prob': oof_prob, 'oof_reg': oof_reg,
        'test_prob': test_prob, 'test_reg': test_reg,
        'mean_fold_score': float(np.mean(fold_scores)),
    }


# -----------------------------------------------------------------------------
# Metric-Driven Late Fusion Grid Search on OOF
# -----------------------------------------------------------------------------
def optimize_fusion_weights(ids, y, pipeline_results, step=0.05):
    names = list(pipeline_results.keys())
    oof_probs = np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1)
    oof_regs = np.stack([pipeline_results[n]['oof_reg'] for n in names], axis=1)
    oof_prios = oof_probs * oof_regs

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    threshold_grid = np.linspace(0.20, 0.80, 25)

    best = {'score': -1.0, 'weights': (1/3, 1/3, 1/3), 'q': 0.0, 'r': 0.0, 'threshold': 0.5}

    print(f"\n{'='*75}\n🔍 Late-Fusion Weight Grid Search (step={step})\n{'='*75}")

    for a in grid:
        for b in grid:
            g = round(1.0 - a - b, 2)
            if g < 0 or g > 1.0:
                continue
            weights = np.array([a, b, g])
            P_ens = oof_probs @ weights
            Prio_ens = oof_prios @ weights

            for t in threshold_grid:
                priority = np.where(P_ens >= t, Prio_ens, 0.0)
                priority = np.clip(priority, 0.0, None)
                sc, q, r = compute_competition_metric(ids, y, priority)
                if sc > best['score']:
                    best.update(score=sc, weights=(a, b, g), q=q, r=r, threshold=float(t))

    print(f"✅ Best weights found: {dict(zip(names, [round(w, 2) for w in best['weights']]))}")
    print(f"   OOF Score={best['score']:.5f} | Q={best['q']:.5f} | R={best['r']:.5f} | T={best['threshold']:.3f}")
    return dict(zip(names, best['weights'])), best


# -----------------------------------------------------------------------------
# Calibrate Test Threshold to Exact Budget (381 papers)
# -----------------------------------------------------------------------------
def calibrate_test_threshold(test_P_ens, target_count=TARGET_TEST_COUNT):
    candidates = np.linspace(0.05, 0.95, 1801)
    best_t, best_diff, best_count = 0.5, 10**9, 0
    for t in candidates:
        cnt = int(np.sum(test_P_ens >= t))
        diff = abs(cnt - target_count)
        if diff < best_diff:
            best_diff, best_t, best_count = diff, float(t), cnt
            if diff == 0:
                break
    return best_t, best_count


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def run_v16_pipeline():
    set_seed(SEED)
    data_dir = os.path.join(BASE_DIR, "data")
    sub_dir = os.path.join(BASE_DIR, "submissions")

    print(f"{'#'*75}\n# V16 MINI-LM CENTRIC TRI-ARCHITECTURE PIPELINE\n"
          f"# (Topic SVD + Sub-Centroids + k-NN Manifold Density), {N_SPLITS}-Fold Stratified CV\n{'#'*75}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    y = train_df['Label'].values.astype(float)
    is_pos = (y >= 2).astype(int)

    # 100% anchored on MiniLM embeddings
    emb_train = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    emb_test = np.load(os.path.join(data_dir, "test_embeddings_minilm.npy"))
    print(f"Loaded pristine MiniLM embeddings: train {emb_train.shape}, test {emb_test.shape}")

    meta_train = extract_clean_metadata(train_df)
    meta_test = extract_clean_metadata(test_df)

    # 1. Run all 3 diverse architectures
    pipeline_results = {}
    for name, spec in PIPELINE_SPECS.items():
        pipeline_results[name] = run_pipeline_cv(
            name, spec, train_df, test_df,
            emb_train, emb_test, y, is_pos, meta_train, meta_test
        )

    # 2. Base Pipeline Comparison Table
    print(f"\n{'='*75}\n📊 BASE ARCHITECTURE OOF COMPARISON\n{'='*75}")
    print(f"{'Pipeline':<18} | {'Score':<8} | {'Q':<8} | {'R':<8} | {'MeanFold':<9}")
    print("-" * 60)
    ids_train = train_df['id'].values
    for name, res in pipeline_results.items():
        priority = np.where(res['oof_prob'] >= 0.5, res['oof_reg'], 0.0)
        priority = np.clip(priority, 0.0, None)
        sc, q, r = compute_competition_metric(ids_train, y, priority)
        print(f"{name:<18} | {sc:<8.5f} | {q:<8.5f} | {r:<8.5f} | {res['mean_fold_score']:<9.5f}")

    # 3. Optimize Late-Fusion Weights
    weights, best_fusion = optimize_fusion_weights(ids_train, y, pipeline_results, step=0.05)

    # 4. Construct Final Test Predictions
    names = list(pipeline_results.keys())
    w = np.array([weights[n] for n in names])
    test_probs = np.stack([pipeline_results[n]['test_prob'] for n in names], axis=1)
    test_regs = np.stack([pipeline_results[n]['test_reg'] for n in names], axis=1)
    test_prios = test_probs * test_regs

    test_P_ens = test_probs @ w
    test_Prio_ens = test_prios @ w

    # 5. Calibrate Test Threshold to exactly 381 papers
    t_star, achieved_count = calibrate_test_threshold(test_P_ens, TARGET_TEST_COUNT)
    print(f"\n{'='*75}\n🎯 THRESHOLD CALIBRATION (TEST SET)\n{'='*75}")
    print(f"   T* = {t_star:.4f} -> {achieved_count} / {len(test_df)} papers "
          f"({100.0 * achieved_count / len(test_df):.2f}%) [target: {TARGET_TEST_COUNT}]")

    test_priorities = np.where(test_P_ens >= t_star, test_Prio_ens, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    # 6. Save Submission File
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v16_minilm_tri_arch.csv",
        output_dir=sub_dir
    )
    print(f"\n[DONE] Saved submission to: {out_file}")

    # 7. Save OOF Audit Trail
    oof_P_ens = np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1) @ w
    oof_Prio_ens = (np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1)
                    * np.stack([pipeline_results[n]['oof_reg'] for n in names], axis=1)) @ w

    oof_df = pd.DataFrame({'id': ids_train, 'Label': y})
    for n in names:
        oof_df[f'oof_prob_{n}'] = pipeline_results[n]['oof_prob']
        oof_df[f'oof_prio_{n}'] = pipeline_results[n]['oof_prob'] * pipeline_results[n]['oof_reg']
    oof_df['oof_prob_ens'] = oof_P_ens
    oof_df['oof_priority_ens'] = oof_Prio_ens

    oof_out_path = os.path.join(sub_dir, "oof_v16_minilm_tri_arch.csv")
    oof_df.to_csv(oof_out_path, index=False)
    print(f"[DONE] Saved OOF audit trail to: {oof_out_path}")

    # 8. Check Overlap with V8/V13 Golden Shortlist
    v13_path = os.path.join(sub_dir, "submission_v13_deep_domain.csv")
    if os.path.exists(v13_path):
        v13 = pd.read_csv(v13_path)
        v13_rec = set(v13[v13['priority'] > 0]['id'])
        v16_rec = set(test_df.iloc[test_priorities > 0]['id'])
        overlap = len(v16_rec.intersection(v13_rec))
        print(f"\n[SHORTLIST INTEGRITY] Overlap with V13 Golden Shortlist: {overlap} / {len(v16_rec)} ({overlap/len(v16_rec)*100:.2f}%)")
        print(f"Correlation with V13 Priorities: {pd.Series(test_priorities).corr(v13['priority']):.5f}")

    print(f"\n{'='*75}\n🔥 V16 MINI-LM TRI-ARCHITECTURE — COMPLETED SUCCESSFULLY!\n{'='*75}\n")


if __name__ == "__main__":
    run_v16_pipeline()
