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

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.metric import compute_competition_metric
from src.utils import set_seed, save_submission

# -----------------------------------------------------------------------------
# V15 Late-Fusion Pipeline
#
# Architecture (per Antigravity's confirmed decision after the V9 1152d early-
# fusion overfitting lesson and the Oracle Simulation headroom analysis):
#   - 3 INDEPENDENT semantic pipelines (MiniLM-384d / MPNet-768d / BGE-384d),
#     each with its own SVD + KMeans/Centroid feature extractor and its own
#     Dual-Head (Stage 1 Classifier + Stage 2 Regressor) LightGBM+CatBoost blend.
#   - 10-Fold Stratified CV (on Label) per pipeline, fully OOF-safe: every
#     stateful transformer (SVD/KMeans/centroids) and every model is fit ONLY
#     on the fold's train split, never on val/test.
#   - Late Fusion: ensemble weights (alpha, beta, gamma) on the 3 pipelines'
#     OOF probability/priority are optimized via grid search directly against
#     compute_competition_metric (no hand-picked ratios).
#   - Final T* threshold calibrated on the TEST set to select exactly 381
#     papers (63.93%), matching the empirically golden zone from V8-V13.
#   - NO author target encoding, NO manual domain-keyword cherry-picking.
# -----------------------------------------------------------------------------

N_SPLITS = 10
SEED = 42
TARGET_TEST_COUNT = 381  # Golden zone count validated by V8-V13 (381/596 = 63.93%)

PROCEEDINGS_KEYWORDS = [
    'proceedings', 'conference on', 'preface', 'workshop on',
    'doctoral consortium', 'front matter', 'editorial'
]


# -----------------------------------------------------------------------------
# Shared clean metadata (NO author/venue target encoding - zero leakage risk)
# -----------------------------------------------------------------------------
def extract_clean_metadata(df: pd.DataFrame) -> np.ndarray:
    """Structural metadata only: year, title length, author count, DOI presence.
    Identical across all 3 pipelines and stateless (no fit required)."""
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
        lambda s: float(any(kw in s for kw in PROCEEDINGS_KEYWORDS)))
    is_proceedings = is_proceedings.values.reshape(-1, 1)

    return np.hstack([
        year_norm, paper_age, is_recent, has_doi,
        author_count, is_single_author, char_len, word_count, is_proceedings
    ])


META_FEATURE_NAMES = [
    'year_norm', 'paper_age', 'is_recent', 'has_doi',
    'author_count', 'is_single_author', 'char_len', 'word_count', 'is_proceedings'
]


# -----------------------------------------------------------------------------
# Pipeline 1: MiniLM (384d) -> SVD(48) + KMeans(12) + 3-way Centroids
# -----------------------------------------------------------------------------
def extract_features_minilm(df, embeddings, meta, is_train, y_train=None, cache=None):
    """SVD(48) + KMeans(12) topic clusters + cosine sim to Pos(>=4)/Top5(==5)/Neg(==1)
    centroids, on top of clean metadata. All stateful transforms fit on train only."""
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        pos_mask = (y_train >= 4)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        pos_c = embeddings[pos_mask].mean(axis=0, keepdims=True)
        top5_c = embeddings[top5_mask].mean(axis=0, keepdims=True)
        neg_c = embeddings[neg_mask].mean(axis=0, keepdims=True)
        pos_c = pos_c / (np.linalg.norm(pos_c) + 1e-9)
        top5_c = top5_c / (np.linalg.norm(top5_c) + 1e-9)
        neg_c = neg_c / (np.linalg.norm(neg_c) + 1e-9)

        svd_model = TruncatedSVD(n_components=48, random_state=SEED)
        svd_feats = svd_model.fit_transform(embeddings)

        kmeans_model = KMeans(n_clusters=12, random_state=SEED, n_init=10)
        cluster_dists = kmeans_model.fit_transform(embeddings)

        cache = {'svd': svd_model, 'kmeans': kmeans_model,
                 'pos_c': pos_c, 'top5_c': top5_c, 'neg_c': neg_c}
    else:
        svd_feats = cache['svd'].transform(embeddings)
        cluster_dists = cache['kmeans'].transform(embeddings)
        pos_c, top5_c, neg_c = cache['pos_c'], cache['top5_c'], cache['neg_c']

    sim_pos = np.dot(embeddings, pos_c.T)
    sim_top5 = np.dot(embeddings, top5_c.T)
    sim_neg = np.dot(embeddings, neg_c.T)
    sim_diff = sim_pos - sim_neg

    X = np.hstack([svd_feats, cluster_dists, sim_pos, sim_top5, sim_neg, sim_diff, meta])
    return X, cache


# -----------------------------------------------------------------------------
# Pipeline 2: MPNet (768d) -> SVD(64) + 8 Sub-Centroids on Pos(>=3) subset
# -----------------------------------------------------------------------------
def extract_features_mpnet(df, embeddings, meta, is_train, y_train=None, cache=None):
    """SVD(64) + 8 K-Means sub-centroids fit on the Label>=3 positive subset
    (captures multi-modal topic structure within 'worthwhile' papers), plus a
    single global negative (Label==1) centroid for contrast."""
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        pos_mask = (y_train >= 3)
        neg_mask = (y_train == 1)

        neg_c = embeddings[neg_mask].mean(axis=0, keepdims=True)
        neg_c = neg_c / (np.linalg.norm(neg_c) + 1e-9)

        km_sub = KMeans(n_clusters=8, random_state=SEED, n_init=10)
        km_sub.fit(embeddings[pos_mask])
        sub_centroids = km_sub.cluster_centers_
        sub_centroids = sub_centroids / (np.linalg.norm(sub_centroids, axis=1, keepdims=True) + 1e-9)

        svd_model = TruncatedSVD(n_components=64, random_state=SEED)
        svd_feats = svd_model.fit_transform(embeddings)

        cache = {'svd': svd_model, 'sub_centroids': sub_centroids, 'neg_c': neg_c}
    else:
        svd_feats = cache['svd'].transform(embeddings)
        sub_centroids = cache['sub_centroids']
        neg_c = cache['neg_c']

    sub_sims = np.dot(embeddings, sub_centroids.T)  # (N, 8)
    max_sub_sim = sub_sims.max(axis=1, keepdims=True)
    mean_sub_sim = sub_sims.mean(axis=1, keepdims=True)
    sim_neg = np.dot(embeddings, neg_c.T)
    sim_diff = max_sub_sim - sim_neg

    X = np.hstack([svd_feats, sub_sims, max_sub_sim, mean_sub_sim, sim_neg, sim_diff, meta])
    return X, cache


# -----------------------------------------------------------------------------
# Pipeline 3: BGE (384d) -> SVD(48) + 3-way Centroids
# -----------------------------------------------------------------------------
def extract_features_bge(df, embeddings, meta, is_train, y_train=None, cache=None):
    """SVD(48) + cosine sim to Pos(>=4)/Top5(==5)/Neg(==1) centroids, on top of
    clean metadata. Mirrors Pipeline 1 but on the BGE retrieval-tuned embedding
    space (no KMeans topic clusters, to keep the 3 pipelines architecturally
    distinct and reduce correlated errors across the ensemble)."""
    if is_train:
        y_train = np.asarray(y_train, dtype=float)
        pos_mask = (y_train >= 4)
        top5_mask = (y_train == 5)
        neg_mask = (y_train == 1)

        pos_c = embeddings[pos_mask].mean(axis=0, keepdims=True)
        top5_c = embeddings[top5_mask].mean(axis=0, keepdims=True)
        neg_c = embeddings[neg_mask].mean(axis=0, keepdims=True)
        pos_c = pos_c / (np.linalg.norm(pos_c) + 1e-9)
        top5_c = top5_c / (np.linalg.norm(top5_c) + 1e-9)
        neg_c = neg_c / (np.linalg.norm(neg_c) + 1e-9)

        svd_model = TruncatedSVD(n_components=48, random_state=SEED)
        svd_feats = svd_model.fit_transform(embeddings)

        cache = {'svd': svd_model, 'pos_c': pos_c, 'top5_c': top5_c, 'neg_c': neg_c}
    else:
        svd_feats = cache['svd'].transform(embeddings)
        pos_c, top5_c, neg_c = cache['pos_c'], cache['top5_c'], cache['neg_c']

    sim_pos = np.dot(embeddings, pos_c.T)
    sim_top5 = np.dot(embeddings, top5_c.T)
    sim_neg = np.dot(embeddings, neg_c.T)
    sim_diff = sim_pos - sim_neg

    X = np.hstack([svd_feats, sim_pos, sim_top5, sim_neg, sim_diff, meta])
    return X, cache


PIPELINE_SPECS = {
    'minilm': {'extractor': extract_features_minilm, 'emb_suffix': 'minilm'},
    'mpnet':  {'extractor': extract_features_mpnet,  'emb_suffix': 'mpnet'},
    'bge':    {'extractor': extract_features_bge,    'emb_suffix': 'bge'},
}


# -----------------------------------------------------------------------------
# Dual-Head model: Stage 1 Classifier (P(is_pos)) + Stage 2 Regressor (y_hat)
# Each head blends LightGBM + CatBoost (anti-overfit hyperparameters).
# -----------------------------------------------------------------------------
def build_dual_head(seed: int):
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=seed, verbose=-1
    )
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24,
        subsample=0.85, colsample_bytree=0.75, random_state=seed, verbose=-1
    )
    heads = {'clf': {'lgb': clf_lgb}, 'reg': {'lgb': reg_lgb}}

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


def fit_dual_head(heads, X_tr, is_pos_tr, y_tr):
    """Fits both heads (LightGBM [+ CatBoost]) ONCE on the fold-train split."""
    for model in heads['clf'].values():
        model.fit(X_tr, is_pos_tr)
    for model in heads['reg'].values():
        model.fit(X_tr, y_tr)
    return heads


def predict_dual_head(heads, X):
    """Blended (prob, reg_pred) on X using already-fitted heads. Equal-weight
    average across available models (LightGBM [+ CatBoost])."""
    clf_probs = [model.predict_proba(X)[:, 1] for model in heads['clf'].values()]
    prob = np.mean(clf_probs, axis=0)

    reg_preds = [model.predict(X) for model in heads['reg'].values()]
    reg = np.mean(reg_preds, axis=0)

    return prob, reg


# -----------------------------------------------------------------------------
# Per-pipeline 10-Fold OOF training (also returns a full-fit model for test)
# -----------------------------------------------------------------------------
def run_pipeline_cv(name, spec, train_df, test_df, emb_train, emb_test, y, is_pos, meta_train, meta_test):
    print(f"\n{'='*70}\n🧩 Pipeline [{name.upper()}] — {N_SPLITS}-Fold Stratified CV\n{'='*70}")
    N = len(train_df)
    oof_prob = np.zeros(N)
    oof_reg = np.zeros(N)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_scores = []

    test_prob_folds = []
    test_reg_folds = []

    extractor = spec['extractor']

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

        heads = build_dual_head(seed=SEED + fold)
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

    # Average test predictions across the 10 fold-fitted models (bagging effect)
    test_prob = np.mean(test_prob_folds, axis=0)
    test_reg = np.mean(test_reg_folds, axis=0)

    print(f"  Mean Fold Score: {np.mean(fold_scores):.5f} (std={np.std(fold_scores):.5f})")

    return {
        'oof_prob': oof_prob, 'oof_reg': oof_reg,
        'test_prob': test_prob, 'test_reg': test_reg,
        'mean_fold_score': float(np.mean(fold_scores)),
    }


# -----------------------------------------------------------------------------
# Late-Fusion weight optimization: grid search on the (alpha, beta, gamma)
# simplex, directly maximizing compute_competition_metric on OOF predictions.
# For each candidate weight triple, P_ens/Prio_ens are built and the decision
# threshold T is ALSO swept on OOF (coarse) so the reported score reflects the
# best achievable score for that weight combo, not a single arbitrary T.
# -----------------------------------------------------------------------------
def optimize_fusion_weights(ids, y, pipeline_results, step=0.05):
    names = list(pipeline_results.keys())
    oof_probs = np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1)
    oof_regs = np.stack([pipeline_results[n]['oof_reg'] for n in names], axis=1)
    oof_prios = oof_probs * oof_regs  # Priority_i = P_i * y_hat_i per spec

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    threshold_grid = np.linspace(0.20, 0.80, 25)

    best = {'score': -1.0, 'weights': (1 / 3, 1 / 3, 1 / 3), 'q': 0.0, 'r': 0.0, 'threshold': 0.5}

    print(f"\n{'='*70}\n🔍 Late-Fusion Weight Grid Search (step={step}, {len(grid)**2} candidate triples)\n{'='*70}")

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

    print(f"✅ Best weights found: {dict(zip(names, [round(w,2) for w in best['weights']]))}")
    print(f"   OOF Score={best['score']:.5f} | Q={best['q']:.5f} | R={best['r']:.5f} | T={best['threshold']:.3f}")
    return dict(zip(names, best['weights'])), best


# -----------------------------------------------------------------------------
# Calibrate test-set threshold T* to hit exactly TARGET_TEST_COUNT papers
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
# Main orchestration
# -----------------------------------------------------------------------------
def run_v15_pipeline():
    set_seed(SEED)
    data_dir = os.path.join(BASE_DIR, "data")
    sub_dir = os.path.join(BASE_DIR, "submissions")

    print(f"{'#'*70}\n# V15 LATE FUSION — Triple Independent Semantic Pipelines\n"
          f"# (MiniLM-384d / MPNet-768d / BGE-384d), {N_SPLITS}-Fold Stratified CV\n{'#'*70}")

    # ---------------- Load data ----------------
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    y = train_df['Label'].values.astype(float)
    is_pos = (y >= 2).astype(int)

    emb_train = {
        'minilm': np.load(os.path.join(data_dir, "train_embeddings_minilm.npy")),
        'mpnet':  np.load(os.path.join(data_dir, "train_embeddings_mpnet.npy")),
        'bge':    np.load(os.path.join(data_dir, "train_embeddings_bge.npy")),
    }
    emb_test = {
        'minilm': np.load(os.path.join(data_dir, "test_embeddings_minilm.npy")),
        'mpnet':  np.load(os.path.join(data_dir, "test_embeddings_mpnet.npy")),
        'bge':    np.load(os.path.join(data_dir, "test_embeddings_bge.npy")),
    }
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    for k in emb_train:
        print(f"  {k}: train {emb_train[k].shape}, test {emb_test[k].shape}")

    meta_train = extract_clean_metadata(train_df)
    meta_test = extract_clean_metadata(test_df)

    # ---------------- Run the 3 independent pipelines ----------------
    pipeline_results = {}
    for name, spec in PIPELINE_SPECS.items():
        pipeline_results[name] = run_pipeline_cv(
            name, spec, train_df, test_df,
            emb_train[name], emb_test[name], y, is_pos, meta_train, meta_test
        )

    # ---------------- Base pipeline OOF comparison table ----------------
    print(f"\n{'='*70}\n📊 BASE PIPELINE OOF COMPARISON\n{'='*70}")
    print(f"{'Pipeline':<10} | {'Score':<8} | {'Q':<8} | {'R':<8} | {'MeanFold':<9}")
    print("-" * 55)
    ids_train = train_df['id'].values
    for name, res in pipeline_results.items():
        priority = np.where(res['oof_prob'] >= 0.5, res['oof_reg'], 0.0)
        priority = np.clip(priority, 0.0, None)
        sc, q, r = compute_competition_metric(ids_train, y, priority)
        print(f"{name:<10} | {sc:<8.5f} | {q:<8.5f} | {r:<8.5f} | {res['mean_fold_score']:<9.5f}")

    # ---------------- Late-Fusion weight optimization ----------------
    weights, best_fusion = optimize_fusion_weights(ids_train, y, pipeline_results, step=0.05)

    # ---------------- Build final ensemble on TEST set ----------------
    names = list(pipeline_results.keys())
    w = np.array([weights[n] for n in names])
    test_probs = np.stack([pipeline_results[n]['test_prob'] for n in names], axis=1)
    test_regs = np.stack([pipeline_results[n]['test_reg'] for n in names], axis=1)
    test_prios = test_probs * test_regs

    test_P_ens = test_probs @ w
    test_Prio_ens = test_prios @ w

    # ---------------- Calibrate T* on TEST set to exactly 381 papers ----------------
    t_star, achieved_count = calibrate_test_threshold(test_P_ens, TARGET_TEST_COUNT)
    print(f"\n{'='*70}\n🎯 THRESHOLD CALIBRATION (TEST SET)\n{'='*70}")
    print(f"   T* = {t_star:.4f} -> {achieved_count} / {len(test_df)} papers "
          f"({100.0 * achieved_count / len(test_df):.2f}%) [target: {TARGET_TEST_COUNT}]")

    test_priorities = np.where(test_P_ens >= t_star, test_Prio_ens, 0.0)
    test_priorities = np.clip(test_priorities, 0.0, None)

    # ---------------- Final summary ----------------
    print(f"\n{'='*70}\n🔥 V15 LATE FUSION — FINAL RESULTS\n{'='*70}")
    print(f"   Fusion weights        : "
          + ", ".join(f"{n}={weights[n]:.2f}" for n in names))
    print(f"   OOF Overall Score     = {best_fusion['score']:.5f}")
    print(f"   Relevance (Q)         = {best_fusion['q']:.5f}")
    print(f"   Coverage (R)          = {best_fusion['r']:.5f}")
    print(f"   OOF threshold used    = {best_fusion['threshold']:.3f}")
    print(f"   Test T* (calibrated)  = {t_star:.4f}")
    print(f"   Test recommended      = {achieved_count} / {len(test_df)} "
          f"({100.0 * achieved_count / len(test_df):.2f}%)")
    print(f"   -- Reference (V13 best Public LB: 0.65071) --")
    print(f"{'='*70}\n")

    # ---------------- Save submission ----------------
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename="submission_v15_late_fusion.csv",
        output_dir=sub_dir
    )
    print(f"[DONE] Saved submission to: {out_file}")

    # ---------------- Save OOF predictions for audit trail ----------------
    oof_P_ens = np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1) @ w
    oof_Prio_ens = (np.stack([pipeline_results[n]['oof_prob'] for n in names], axis=1)
                    * np.stack([pipeline_results[n]['oof_reg'] for n in names], axis=1)) @ w

    oof_df = pd.DataFrame({'id': ids_train, 'Label': y})
    for n in names:
        oof_df[f'oof_prob_{n}'] = pipeline_results[n]['oof_prob']
        oof_df[f'oof_prio_{n}'] = pipeline_results[n]['oof_prob'] * pipeline_results[n]['oof_reg']
    oof_df['oof_prob_ens'] = oof_P_ens
    oof_df['oof_priority_ens'] = oof_Prio_ens

    oof_out_path = os.path.join(sub_dir, "oof_v15_late_fusion.csv")
    oof_df.to_csv(oof_out_path, index=False)
    print(f"[DONE] Saved OOF audit trail to: {oof_out_path}")

    return {
        'weights': weights,
        'oof_score': best_fusion['score'],
        'oof_q': best_fusion['q'],
        'oof_r': best_fusion['r'],
        'test_threshold': t_star,
        'test_recom_count': achieved_count,
        'submission_file': out_file,
        'oof_file': oof_out_path,
    }


if __name__ == "__main__":
    run_v15_pipeline()


