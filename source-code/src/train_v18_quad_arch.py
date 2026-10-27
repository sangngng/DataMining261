import os
import sys

# Ensure source-code root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
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
# V18 QUAD-ARCHITECTURE TRANSDUCTIVE PIPELINE
#
# Formulated based on Gemini Argon's Grandmaster Strategic Blueprint:
#   1. Pure MiniLM Embedding Geometry (Zero Semantic Drift):
#      All representations are strictly derived from all-MiniLM-L6-v2 (384d).
#   2. Four Diverse Inductive & Transductive Paradigms:
#      - P1 (Global Topic SVD & Centroids): SVD(48) + KMeans(12) +
#        Preference Centroids (LGBM + CatBoost).
#      - P2 (Positive Sub-Centroids): SVD(64) + K=8 sub-centroids on Label>=3
#        + Margin (XGBoost + CatBoost).
#      - P3 (Manifold k-NN Density): SVD(48) + k-NN density (k=5, 15) +
#        Nearest Pos/Neg Margin + Bayesian Venue (LGBM + XGBoost).
#      - P4 (Transductive Spectral Graph + Hard-Negative Boundary):
#        Multi-scale spectral diffusion across 3,090 papers (Train + Test)
#        over MiniLM top-15 mutual k-NN and author Jaccard overlap +
#        Hard-negative boundary contrastive centroids (LGBM + CatBoost).
#   3. Strict 10-Fold Stratified Cross-Validation:
#      Label propagation and features for validation nodes are 100% OOF-safe
#      (val nodes receive zero initial labels).
#   4. Decoupled Selection-Ranking Consensus:
#      - Selection (Stage 1): Locked to the 381 candidate shortlist.
#      - Ranking (Stage 2): Re-ranked using rank-normalized Quad-Arch +
#        V17 top-rank DCG + consensus disagreement penalty.
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
# Pipeline 1: Global Topic & Centroids (LightGBM + CatBoost)
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
# Pipeline 2: Multi-Modal Sub-Centroids on Positives (XGBoost + CatBoost)
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
# Pipeline 3: Manifold k-NN Density & Margin (LightGBM + XGBoost)
# -----------------------------------------------------------------------------
def extract_features_p3(df, emb, meta, is_train, y_train=None, cache=None):
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        
        svd_model = TruncatedSVD(n_components=48, random_state=SEED)
        svd_feats = svd_model.fit_transform(emb)

        nn_model = NearestNeighbors(n_neighbors=21, metric='cosine', algorithm='brute')
        nn_model.fit(emb)

        pos_exemplars = emb[y_train >= 4]
        neg_exemplars = emb[y_train == 1]
        nn_pos = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute').fit(pos_exemplars)
        nn_neg = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute').fit(neg_exemplars)

        global_mean = float(np.mean(y_train > 1.0))
        venue_df = pd.DataFrame({'venue': df['venue'], 'is_pos': (y_train > 1.0).astype(float)})
        venue_stats = venue_df.groupby('venue')['is_pos'].agg(['count', 'mean'])
        m = 5.0
        venue_bayes = (venue_stats['count'] * venue_stats['mean'] + m * global_mean) / (venue_stats['count'] + m)
        venue_map = venue_bayes.to_dict()

        cache = {
            'svd': svd_model, 'nn': nn_model, 'nn_pos': nn_pos, 'nn_neg': nn_neg,
            'y_train': y_train, 'venue_map': venue_map, 'global_mean': global_mean
        }
        
        dists, indices = nn_model.kneighbors(emb, n_neighbors=21)
        indices = indices[:, 1:]
        dists = dists[:, 1:]
    else:
        svd_feats = cache['svd'].transform(emb)
        dists, indices = cache['nn'].kneighbors(emb, n_neighbors=20)
        y_train = cache['y_train']

    nbr_labels = y_train[indices]
    pos_ratio_5 = (nbr_labels[:, :5] >= 2).mean(axis=1, keepdims=True)
    pos_ratio_15 = (nbr_labels[:, :15] >= 2).mean(axis=1, keepdims=True)
    top_ratio_5 = (nbr_labels[:, :5] >= 4).mean(axis=1, keepdims=True)
    top_ratio_15 = (nbr_labels[:, :15] >= 4).mean(axis=1, keepdims=True)
    neg_ratio_5 = (nbr_labels[:, :5] == 1).mean(axis=1, keepdims=True)
    local_mean_5 = (nbr_labels[:, :5].mean(axis=1, keepdims=True) - 1.0) / 4.0
    local_mean_15 = (nbr_labels[:, :15].mean(axis=1, keepdims=True) - 1.0) / 4.0
    mean_dist_5 = dists[:, :5].mean(axis=1, keepdims=True)

    d_pos, _ = cache['nn_pos'].kneighbors(emb)
    d_neg, _ = cache['nn_neg'].kneighbors(emb)
    margin_exemplar = d_neg - d_pos

    venue_feat = df['venue'].map(cache['venue_map']).fillna(cache['global_mean']).values.reshape(-1, 1)

    X = np.hstack([
        svd_feats, pos_ratio_5, pos_ratio_15, top_ratio_5, top_ratio_15,
        neg_ratio_5, local_mean_5, local_mean_15, mean_dist_5,
        d_pos, d_neg, margin_exemplar, venue_feat, meta
    ])
    return X, cache


# -----------------------------------------------------------------------------
# Pipeline 4: Transductive Spectral Diffusion + Boundary Contrast (Argon P4)
# -----------------------------------------------------------------------------
def build_transductive_graph(train_emb, test_emb, train_df, test_df):
    """Constructs the 3090 x 3090 symmetric normalized transition matrix S."""
    print("  [P4] Constructing 3,090-node Transductive Affinity Matrix (Semantic + Author-Jaccard)...")
    all_emb = np.vstack([train_emb, test_emb])
    n_total = len(all_emb)

    # 1. Top-15 Mutual k-NN in MiniLM Cosine Space
    sim_matrix = cosine_similarity(all_emb)
    np.fill_diagonal(sim_matrix, -1.0)
    K = 15
    tau = 0.08
    top_k_idx = np.argpartition(sim_matrix, -K, axis=1)[:, -K:]
    rows = np.arange(n_total)[:, None]
    W_sem = np.zeros_like(sim_matrix, dtype=np.float32)
    W_sem[rows, top_k_idx] = np.exp((sim_matrix[rows, top_k_idx] - 1.0) / tau)
    W_sem = 0.5 * (W_sem + W_sem.T)

    # 2. Unsupervised Author-Set Jaccard Overlap
    all_authors = pd.concat([train_df["authors"], test_df["authors"]], ignore_index=True).fillna("")
    author_sets = [set(a.lower().strip() for a in str(s).split(",") if len(a.strip()) > 2) for s in all_authors]
    
    author_to_papers = {}
    for idx, aset in enumerate(author_sets):
        for auth in aset:
            author_to_papers.setdefault(auth, []).append(idx)

    W_auth = np.zeros((n_total, n_total), dtype=np.float32)
    for auth, p_list in author_to_papers.items():
        if 1 < len(p_list) <= 80:
            for i in range(len(p_list)):
                u = p_list[i]
                for j in range(i + 1, len(p_list)):
                    v = p_list[j]
                    inter = len(author_sets[u] & author_sets[v])
                    union = len(author_sets[u] | author_sets[v])
                    jacc = inter / union
                    W_auth[u, v] = max(W_auth[u, v], jacc)
                    W_auth[v, u] = W_auth[u, v]

    W = W_sem + 0.30 * W_auth
    deg = np.sum(W, axis=1) + 1e-8
    inv_sqrt_deg = 1.0 / np.sqrt(deg)
    S = W * inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]

    transductive_deg = np.sum(W_sem > 0, axis=1, keepdims=True)
    auth_bridge_deg = np.sum(W_auth > 0, axis=1, keepdims=True)
    centrality_feats = np.hstack([transductive_deg / 30.0, auth_bridge_deg / 10.0])

    print("  [P4] Transductive Transition Matrix S built successfully!")
    return S, centrality_feats


def extract_features_p4_transductive(
    df, emb, meta, is_train, y_train=None, cache=None,
    S=None, centrality_feats=None, train_idx=None, val_idx=None, n_train=2494
):
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        n_total = len(S)
        
        # OOF-Safe Prior-Centered Label Matrix (3090 x 5)
        # Columns: [y>=2, y>=3, y>=4, y==5, continuous_gain]
        Y = np.zeros((n_total, 5), dtype=np.float32)
        priors = [
            np.mean(y_train >= 2),
            np.mean(y_train >= 3),
            np.mean(y_train >= 4),
            np.mean(y_train == 5),
            np.mean((y_train - 1.0) / 4.0)
        ]
        Y[train_idx, 0] = (y_train >= 2).astype(np.float32) - priors[0]
        Y[train_idx, 1] = (y_train >= 3).astype(np.float32) - priors[1]
        Y[train_idx, 2] = (y_train >= 4).astype(np.float32) - priors[2]
        Y[train_idx, 3] = (y_train == 5).astype(np.float32) - priors[3]
        Y[train_idx, 4] = ((y_train - 1.0) / 4.0).astype(np.float32) - priors[4]

        # Multi-scale Spectral Diffusion
        F_1hop = S @ Y
        F_2hop = S @ F_1hop
        F_ppr  = 0.45 * F_1hop + 0.30 * F_2hop + 0.25 * (S @ F_2hop)

        # Contrastive Hard Boundary Centroids
        pos_core = emb[y_train >= 4]
        pos_border = emb[np.isin(y_train, [2, 3])]
        neg_core = emb[y_train == 1]
        
        c_core = pos_core.mean(axis=0, keepdims=True)
        c_border = pos_border.mean(axis=0, keepdims=True)
        c_neg = neg_core.mean(axis=0, keepdims=True)
        c_core = c_core / (np.linalg.norm(c_core) + 1e-9)
        c_border = c_border / (np.linalg.norm(c_border) + 1e-9)
        c_neg = c_neg / (np.linalg.norm(c_neg) + 1e-9)

        svd_model = TruncatedSVD(n_components=48, random_state=SEED)
        svd_feats = svd_model.fit_transform(emb)

        cache = {
            'svd': svd_model, 'c_core': c_core, 'c_border': c_border, 'c_neg': c_neg,
            'F_1hop_test': F_1hop[n_train:], 'F_2hop_test': F_2hop[n_train:],
            'F_ppr_test': F_ppr[n_train:], 'centrality_test': centrality_feats[n_train:]
        }

        diff_feats_tr = np.hstack([F_1hop[train_idx], F_2hop[train_idx], F_ppr[train_idx], centrality_feats[train_idx]])
        diff_feats_val = np.hstack([F_1hop[val_idx], F_2hop[val_idx], F_ppr[val_idx], centrality_feats[val_idx]])

        sim_c = np.dot(emb, c_core.T)
        sim_b = np.dot(emb, c_border.T)
        sim_n = np.dot(emb, c_neg.T)
        margin_b = sim_b - sim_n
        margin_c = sim_c - sim_n

        contrast_feats = np.hstack([sim_c, sim_b, sim_n, margin_b, margin_c])
        X_tr = np.hstack([svd_feats, diff_feats_tr, contrast_feats, meta])
        return X_tr, cache, diff_feats_val
    else:
        svd_feats = cache['svd'].transform(emb)
        sim_c = np.dot(emb, cache['c_core'].T)
        sim_b = np.dot(emb, cache['c_border'].T)
        sim_n = np.dot(emb, cache['c_neg'].T)
        margin_b = sim_b - sim_n
        margin_c = sim_c - sim_n
        contrast_feats = np.hstack([sim_c, sim_b, sim_n, margin_b, margin_c])

        diff_feats_test = np.hstack([cache['F_1hop_test'], cache['F_2hop_test'], cache['F_ppr_test'], cache['centrality_test']])
        X_test = np.hstack([svd_feats, diff_feats_test, contrast_feats, meta])
        return X_test, cache


# -----------------------------------------------------------------------------
# Dual-Head Model Builders
# -----------------------------------------------------------------------------
def build_dual_head_p1(seed: int):
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
    clf_xgb = xgb.XGBClassifier(
        n_estimators=220, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.75, random_state=seed,
        eval_metric='logloss', verbosity=0
    )
    reg_xgb = xgb.XGBRegressor(
        n_estimators=220, learning_rate=0.03, max_depth=4,
        subsample=0.85, colsample_bytree=0.75, random_state=seed, verbosity=0
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
        subsample=0.80, colsample_bytree=0.70, random_state=seed, verbosity=0
    )
    return {
        'clf': {'lgb': clf_lgb, 'xgb': clf_xgb},
        'reg': {'lgb': reg_lgb, 'xgb': reg_xgb}
    }


def build_dual_head_p4(seed: int):
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=220, learning_rate=0.03, max_depth=4, num_leaves=20,
        subsample=0.85, colsample_bytree=0.70, random_state=seed, verbose=-1
    )
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=220, learning_rate=0.03, max_depth=4, num_leaves=20,
        subsample=0.85, colsample_bytree=0.70, random_state=seed, verbose=-1
    )
    heads = {'clf': {'lgb': clf_lgb}, 'reg': {'lgb': reg_lgb}}
    if CATBOOST_AVAILABLE:
        heads['clf']['cat'] = CatBoostClassifier(
            iterations=220, learning_rate=0.03, depth=5,
            subsample=0.85, colsample_bylevel=0.70, bootstrap_type='Bernoulli',
            random_seed=seed, verbose=False
        )
        heads['reg']['cat'] = CatBoostRegressor(
            iterations=220, learning_rate=0.03, depth=5,
            subsample=0.85, colsample_bylevel=0.70, bootstrap_type='Bernoulli',
            random_seed=seed, verbose=False
        )
    return heads


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
# 10-Fold Stratified Cross-Validation Execution
# -----------------------------------------------------------------------------
def run_all_quad_cv(train_df, test_df, emb_train, emb_test, y, is_pos, meta_train, meta_test):
    N = len(train_df)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    pipeline_names = ['p1_topic', 'p2_subcluster', 'p3_density', 'p4_transductive']
    pipeline_results = {
        name: {
            'oof_prob': np.zeros(N), 'oof_reg': np.zeros(N),
            'test_prob_folds': [], 'test_reg_folds': [], 'fold_scores': []
        }
        for name in pipeline_names
    }

    # Pre-build transductive graph matrix once (unsupervised)
    S_graph, centrality_feats = build_transductive_graph(emb_train, emb_test, train_df, test_df)

    print(f"\n{'='*75}\n🚀 RUNNING 10-FOLD STRATIFIED CV ACROSS ALL 4 ARCHITECTURES\n{'='*75}")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
        y_tr, y_val = y[tr_idx], y[val_idx]
        is_pos_tr = is_pos[tr_idx]
        ids_val = train_df['id'].values[val_idx]

        print(f"\n--- FOLD {fold+1:2d}/{N_SPLITS} ---")

        # P1: Topic SVD & Centroids
        X_tr_p1, cache_p1 = extract_features_p1(train_df.iloc[tr_idx], emb_train[tr_idx], meta_train[tr_idx], is_train=True, y_train=y_tr)
        X_val_p1, _ = extract_features_p1(train_df.iloc[val_idx], emb_train[val_idx], meta_train[val_idx], is_train=False, cache=cache_p1)
        X_te_p1, _ = extract_features_p1(test_df, emb_test, meta_test, is_train=False, cache=cache_p1)
        h1 = build_dual_head_p1(seed=SEED + fold)
        fit_dual_head(h1, X_tr_p1, is_pos_tr, y_tr)
        prob_val_1, reg_val_1 = predict_dual_head(h1, X_val_p1)
        prob_te_1, reg_te_1 = predict_dual_head(h1, X_te_p1)
        pipeline_results['p1_topic']['oof_prob'][val_idx] = prob_val_1
        pipeline_results['p1_topic']['oof_reg'][val_idx] = reg_val_1
        pipeline_results['p1_topic']['test_prob_folds'].append(prob_te_1)
        pipeline_results['p1_topic']['test_reg_folds'].append(reg_te_1)
        sc1, q1, r1 = compute_competition_metric(ids_val, y_val, np.where(prob_val_1 >= 0.5, reg_val_1, 0.0))
        pipeline_results['p1_topic']['fold_scores'].append(sc1)

        # P2: Sub-Centroids
        X_tr_p2, cache_p2 = extract_features_p2(train_df.iloc[tr_idx], emb_train[tr_idx], meta_train[tr_idx], is_train=True, y_train=y_tr)
        X_val_p2, _ = extract_features_p2(train_df.iloc[val_idx], emb_train[val_idx], meta_train[val_idx], is_train=False, cache=cache_p2)
        X_te_p2, _ = extract_features_p2(test_df, emb_test, meta_test, is_train=False, cache=cache_p2)
        h2 = build_dual_head_p2(seed=SEED + fold)
        fit_dual_head(h2, X_tr_p2, is_pos_tr, y_tr)
        prob_val_2, reg_val_2 = predict_dual_head(h2, X_val_p2)
        prob_te_2, reg_te_2 = predict_dual_head(h2, X_te_p2)
        pipeline_results['p2_subcluster']['oof_prob'][val_idx] = prob_val_2
        pipeline_results['p2_subcluster']['oof_reg'][val_idx] = reg_val_2
        pipeline_results['p2_subcluster']['test_prob_folds'].append(prob_te_2)
        pipeline_results['p2_subcluster']['test_reg_folds'].append(reg_te_2)
        sc2, q2, r2 = compute_competition_metric(ids_val, y_val, np.where(prob_val_2 >= 0.5, reg_val_2, 0.0))
        pipeline_results['p2_subcluster']['fold_scores'].append(sc2)

        # P3: Manifold k-NN Density
        X_tr_p3, cache_p3 = extract_features_p3(train_df.iloc[tr_idx], emb_train[tr_idx], meta_train[tr_idx], is_train=True, y_train=y_tr)
        X_val_p3, _ = extract_features_p3(train_df.iloc[val_idx], emb_train[val_idx], meta_train[val_idx], is_train=False, cache=cache_p3)
        X_te_p3, _ = extract_features_p3(test_df, emb_test, meta_test, is_train=False, cache=cache_p3)
        h3 = build_dual_head_p3(seed=SEED + fold)
        fit_dual_head(h3, X_tr_p3, is_pos_tr, y_tr)
        prob_val_3, reg_val_3 = predict_dual_head(h3, X_val_p3)
        prob_te_3, reg_te_3 = predict_dual_head(h3, X_te_p3)
        pipeline_results['p3_density']['oof_prob'][val_idx] = prob_val_3
        pipeline_results['p3_density']['oof_reg'][val_idx] = reg_val_3
        pipeline_results['p3_density']['test_prob_folds'].append(prob_te_3)
        pipeline_results['p3_density']['test_reg_folds'].append(reg_te_3)
        sc3, q3, r3 = compute_competition_metric(ids_val, y_val, np.where(prob_val_3 >= 0.5, reg_val_3, 0.0))
        pipeline_results['p3_density']['fold_scores'].append(sc3)

        # P4: Transductive Graph & Hard-Boundary Contrast
        X_tr_p4, cache_p4, diff_val = extract_features_p4_transductive(
            train_df.iloc[tr_idx], emb_train[tr_idx], meta_train[tr_idx], is_train=True, y_train=y_tr,
            S=S_graph, centrality_feats=centrality_feats, train_idx=tr_idx, val_idx=val_idx, n_train=N
        )
        # Construct val features
        svd_val_p4 = cache_p4['svd'].transform(emb_train[val_idx])
        sim_c_v = np.dot(emb_train[val_idx], cache_p4['c_core'].T)
        sim_b_v = np.dot(emb_train[val_idx], cache_p4['c_border'].T)
        sim_n_v = np.dot(emb_train[val_idx], cache_p4['c_neg'].T)
        contrast_val = np.hstack([sim_c_v, sim_b_v, sim_n_v, sim_b_v - sim_n_v, sim_c_v - sim_n_v])
        X_val_p4 = np.hstack([svd_val_p4, diff_val, contrast_val, meta_train[val_idx]])

        X_te_p4, _ = extract_features_p4_transductive(
            test_df, emb_test, meta_test, is_train=False, cache=cache_p4
        )
        h4 = build_dual_head_p4(seed=SEED + fold)
        fit_dual_head(h4, X_tr_p4, is_pos_tr, y_tr)
        prob_val_4, reg_val_4 = predict_dual_head(h4, X_val_p4)
        prob_te_4, reg_te_4 = predict_dual_head(h4, X_te_p4)
        pipeline_results['p4_transductive']['oof_prob'][val_idx] = prob_val_4
        pipeline_results['p4_transductive']['oof_reg'][val_idx] = reg_val_4
        pipeline_results['p4_transductive']['test_prob_folds'].append(prob_te_4)
        pipeline_results['p4_transductive']['test_reg_folds'].append(reg_te_4)
        sc4, q4, r4 = compute_competition_metric(ids_val, y_val, np.where(prob_val_4 >= 0.5, reg_val_4, 0.0))
        pipeline_results['p4_transductive']['fold_scores'].append(sc4)

        print(f"  P1 Score: {sc1:.5f} | P2 Score: {sc2:.5f} | P3 Score: {sc3:.5f} | P4 Score: {sc4:.5f}")

    # Bagged test predictions
    for name in pipeline_names:
        pipeline_results[name]['test_prob'] = np.mean(pipeline_results[name]['test_prob_folds'], axis=0)
        pipeline_results[name]['test_reg'] = np.mean(pipeline_results[name]['test_reg_folds'], axis=0)

    return pipeline_results


# -----------------------------------------------------------------------------
# Metric-Driven Late Fusion Grid Search on OOF (4-Weights Simplex)
# -----------------------------------------------------------------------------
def optimize_quad_fusion_weights(ids, y, pipeline_results, step=0.05):
    names = ['p1_topic', 'p2_subcluster', 'p3_density', 'p4_transductive']
    oof_probs = np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1)
    oof_regs = np.stack([pipeline_results[n]['oof_reg'] for n in names], axis=1)
    oof_prios = oof_probs * oof_regs

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    threshold_grid = np.linspace(0.20, 0.80, 25)

    best = {'score': -1.0, 'weights': (0.25, 0.25, 0.25, 0.25), 'q': 0.0, 'r': 0.0, 'threshold': 0.5}

    print(f"\n{'='*75}\n🔍 Quad-Fusion Weight Grid Search (step={step})\n{'='*75}")

    for a in grid:
        for b in grid:
            if a + b > 1.0:
                continue
            for c in grid:
                d = round(1.0 - a - b - c, 2)
                if d < 0 or d > 1.0:
                    continue
                weights = np.array([a, b, c, d])
                P_ens = oof_probs @ weights
                Prio_ens = oof_prios @ weights

                for t in threshold_grid:
                    priority = np.where(P_ens >= t, Prio_ens, 0.0)
                    priority = np.clip(priority, 0.0, None)
                    sc, q, r = compute_competition_metric(ids, y, priority)
                    if sc > best['score']:
                        best.update(score=sc, weights=(a, b, c, d), q=q, r=r, threshold=float(t))

    print(f"✅ Best weights found: {dict(zip(names, [round(w, 2) for w in best['weights']]))}")
    print(f"   OOF Score={best['score']:.5f} | Q={best['q']:.5f} | R={best['r']:.5f} | T={best['threshold']:.3f}")
    return dict(zip(names, best['weights'])), best


# -----------------------------------------------------------------------------
# Calibrate Test Threshold to Exactly 381 Papers
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
# Decoupled Stage 1 + Stage 2 Consensus Ranking
# -----------------------------------------------------------------------------
def apply_decoupled_consensus(sub_quad_raw, v17_path, test_df):
    """Combines Quad-Arch high-recall selection with V17 sharp top-50 DCG ranking."""
    v17 = pd.read_csv(v17_path).sort_values('id').reset_index(drop=True)
    v18 = sub_quad_raw.sort_values('id').reset_index(drop=True)

    p18 = v18['priority'].values.copy()
    p17 = v17['priority'].values.copy()

    # Shortlist is strictly V18's 381 candidates
    sel_idx = np.where(p18 > 0)[0]
    assert len(sel_idx) == TARGET_TEST_COUNT

    # Rank-normalize V18 among selected
    s18_raw = p18[sel_idx]
    s18_minmax = (s18_raw - s18_raw.min()) / (s18_raw.max() - s18_raw.min() + 1e-12)
    s18_rank = rankdata(s18_raw) / len(sel_idx)
    score_18 = 0.60 * s18_minmax + 0.40 * s18_rank

    # Rank-normalize V17 among selected
    s17_raw = p17[sel_idx]
    s17_pos = s17_raw > 0
    s17_norm = np.zeros_like(s17_raw)
    if s17_pos.sum() > 0:
        pos_vals = s17_raw[s17_pos]
        pos_minmax = (pos_vals - pos_vals.min()) / (pos_vals.max() - pos_vals.min() + 1e-12)
        pos_rank = rankdata(pos_vals) / len(pos_vals)
        s17_norm[s17_pos] = 0.65 * pos_minmax + 0.35 * pos_rank
    s17_norm[~s17_pos] = 0.85 * score_18[~s17_pos]

    disagreement = np.abs(score_18 - s17_norm)
    blended_prio = 0.65 * score_18 + 0.35 * s17_norm - 0.12 * disagreement
    blended_prio = blended_prio - blended_prio.min() + 0.10

    final_prio = np.zeros(len(v18), dtype=np.float64)
    final_prio[sel_idx] = blended_prio

    sub_decoupled = pd.DataFrame({'id': v18['id'], 'priority': np.round(final_prio, 6)})
    return sub_decoupled


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def run_v18_pipeline():
    set_seed(SEED)
    data_dir = os.path.join(BASE_DIR, "data")
    sub_dir = os.path.join(BASE_DIR, "submissions")

    print(f"{'#'*75}\n# V18 QUAD-ARCHITECTURE TRANSDUCTIVE PIPELINE\n"
          f"# (Topic SVD + Sub-Centroids + k-NN Density + Transductive Spectral Diffusion)\n{'#'*75}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    y = train_df['Label'].values.astype(float)
    is_pos = (y >= 2).astype(int)

    emb_train = np.load(os.path.join(data_dir, "train_embeddings_minilm.npy"))
    emb_test = np.load(os.path.join(data_dir, "test_embeddings_minilm.npy"))
    print(f"Loaded pristine MiniLM embeddings: train {emb_train.shape}, test {emb_test.shape}")

    meta_train = extract_clean_metadata(train_df)
    meta_test = extract_clean_metadata(test_df)

    # 1. Run 10-Fold CV across all 4 architectures
    pipeline_results = run_all_quad_cv(
        train_df, test_df, emb_train, emb_test, y, is_pos, meta_train, meta_test
    )

    # 2. Base Pipeline Comparison Table
    print(f"\n{'='*75}\n📊 QUAD-ARCHITECTURE BASE OOF COMPARISON\n{'='*75}")
    print(f"{'Pipeline':<18} | {'Score':<8} | {'Q':<8} | {'R':<8} | {'MeanFold':<9}")
    print("-" * 60)
    ids_train = train_df['id'].values
    for name, res in pipeline_results.items():
        priority = np.where(res['oof_prob'] >= 0.5, res['oof_reg'], 0.0)
        priority = np.clip(priority, 0.0, None)
        sc, q, r = compute_competition_metric(ids_train, y, priority)
        m_fold = float(np.mean(res['fold_scores']))
        print(f"{name:<18} | {sc:<8.5f} | {q:<8.5f} | {r:<8.5f} | {m_fold:<9.5f}")

    # 3. Optimize Late-Fusion Weights
    weights, best_fusion = optimize_quad_fusion_weights(ids_train, y, pipeline_results, step=0.05)

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

    # 6. Save Raw Quad-Arch Submission
    raw_sub_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v18_quad_arch_raw.csv",
        output_dir=sub_dir
    )
    print(f"\n[DONE] Saved Raw Quad-Arch Submission to: {raw_sub_file}")

    # 7. Apply Decoupled Consensus Ranking with V17
    v17_path = os.path.join(sub_dir, "submission_v17_decoupled_consensus.csv")
    sub_raw_df = pd.read_csv(raw_sub_file)
    sub_decoupled = apply_decoupled_consensus(sub_raw_df, v17_path, test_df)
    
    decoupled_sub_file = os.path.join(sub_dir, "submission_v18_decoupled_consensus.csv")
    sub_decoupled.to_csv(decoupled_sub_file, index=False)
    print(f"[DONE] Saved Decoupled Consensus Submission to: {decoupled_sub_file}")

    # 8. Save OOF Audit Trail
    oof_P_ens = np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1) @ w
    oof_Prio_ens = (np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1)
                    * np.stack([pipeline_results[n]['oof_reg'] for n in names], axis=1)) @ w

    oof_df = pd.DataFrame({'id': ids_train, 'Label': y})
    for n in names:
        oof_df[f'oof_prob_{n}'] = pipeline_results[n]['oof_prob']
        oof_df[f'oof_prio_{n}'] = pipeline_results[n]['oof_prob'] * pipeline_results[n]['oof_reg']
    oof_df['oof_prob_ens'] = oof_P_ens
    oof_df['oof_priority_ens'] = oof_Prio_ens

    oof_out_path = os.path.join(sub_dir, "oof_v18_quad_arch.csv")
    oof_df.to_csv(oof_out_path, index=False)
    print(f"[DONE] Saved OOF audit trail to: {oof_out_path}")

    # 9. Verify Shortlist Overlap with V17 / V16
    v17_rec = set(pd.read_csv(v17_path).query("priority > 0")['id'])
    v18_rec = set(sub_decoupled.query("priority > 0")['id'])
    overlap_17 = len(v18_rec.intersection(v17_rec))
    print(f"\n[SHORTLIST INTEGRITY] Overlap with V17 Golden Shortlist: {overlap_17} / {len(v18_rec)} ({overlap_17/len(v18_rec)*100:.2f}%)")
    print(f"Correlation with V17 Priorities: {sub_decoupled['priority'].corr(pd.read_csv(v17_path)['priority']):.5f}")

    print(f"\n{'='*75}\n🔥 V18 QUAD-ARCHITECTURE TRANSDUCTIVE PIPELINE — COMPLETED SUCCESSFULLY!\n{'='*75}\n")


if __name__ == "__main__":
    run_v18_pipeline()
