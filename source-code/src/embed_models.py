"""Title-embedding learners (pretrained sentence encoders on the title only; no labels used to build the vectors).

Representation: all-MiniLM-L6-v2 (Reimers & Gurevych, 2019) and BAAI/bge-base-en-v1.5 (Xiao et al., 2023),
computed by src/data_prep/make_specter_embeddings.py (or colab_make_embeddings.py) and stored as
data/emb_{minilm,bge}_title_{train,test}.npy in the row order of train/test_with_abstracts.csv.

Learner (classification + regression on kNN / prototype features):
  * kNN label statistics among the 5 / 15 most similar labelled papers,
  * similarity to the nearest liked (y>=4) and disliked (y=1) paper and their margin,
  * similarity to the centroids of liked / disliked / y=5 papers,
  * 48-d SVD of the embedding + venue code, year, #authors, title length,
  -> LightGBM regressor of the gain (y-1)/4 and LightGBM classifier of P(y>1), 3 seeds each.
All label-dependent quantities use the training rows `tr` only; a training row never sees itself.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from .data import DATA, SEED

TAGS = ["minilm_title", "bge_title"]
LGB_P = dict(n_estimators=220, learning_rate=0.03, num_leaves=15, min_child_samples=25, subsample=0.85,
             subsample_freq=1, colsample_bytree=0.75, verbose=-1, n_jobs=2)


def load_title_embeddings():
    out = {}
    for t in TAGS:
        a, b = np.load(DATA / f"emb_{t}_train.npy"), np.load(DATA / f"emb_{t}_test.npy")
        E = np.vstack([a, b]).astype(np.float32)
        out[t] = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9), TruncatedSVD(48, random_state=SEED).fit_transform(E))
    return out


def _feats(E, tr, ytr, rows, self_ex):
    sim = E[rows] @ E[tr].T
    if self_ex:
        sim[np.arange(len(rows)), np.arange(len(rows))] = -9.0
    o = np.argsort(-sim, 1)[:, :15]; L = ytr[o]
    pos, neg = ytr >= 4, ytr == 1
    cp, cn, c5 = E[tr][pos].mean(0), E[tr][neg].mean(0), E[tr][ytr == 5].mean(0)
    dpos = np.where(pos[None, :], sim, -9).max(1); dneg = np.where(neg[None, :], sim, -9).max(1)
    return np.column_stack([(L[:, :5] > 1).mean(1), (L > 1).mean(1), (L[:, :5] >= 4).mean(1), (L >= 4).mean(1),
                            (L[:, :5] - 1).mean(1) / 4, (L - 1).mean(1) / 4, np.take_along_axis(sim, o[:, :5], 1).mean(1),
                            dpos, dneg, dpos - dneg, E[rows] @ cp, E[rows] @ cn, E[rows] @ c5, E[rows] @ (cp - cn)])


def title_emb_predict(S, E, svd, tr, ytr, va):
    """Returns (expected-gain regression, P(y>1)) for rows `va`, fitted on rows `tr` only."""
    meta = lambda r: np.column_stack([svd[r], S.vcodes[r], S.year[r], S.nauth[r], S.tl[r]])
    Ftr = np.hstack([_feats(E, tr, ytr, tr, True), meta(tr)]); Fva = np.hstack([_feats(E, tr, ytr, va, False), meta(va)])
    reg = np.mean([lgb.LGBMRegressor(**LGB_P, random_state=SEED + s).fit(Ftr, (ytr - 1) / 4).predict(Fva) for s in range(3)], 0)
    clf = np.mean([lgb.LGBMClassifier(**LGB_P, random_state=SEED + s).fit(Ftr, ytr > 1).predict_proba(Fva)[:, 1] for s in range(3)], 0)
    return reg, clf


def title_emb_utility(S, embs, tr, ytr, va):
    """Mean rank of (regression, classifier) per encoder -> one utility per encoder, list of length len(TAGS)."""
    from .models import rank01
    out = []
    for t in TAGS:
        r, c = title_emb_predict(S, *embs[t], tr, ytr, va)
        out.append((rank01(r) + rank01(c)) / 2)
    return out
