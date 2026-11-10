"""Base learners (classification / regression / clustering / itemset rules / kNN) and the stacker.
Every label-dependent quantity is fitted only on the rows passed as `tr`."""
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from .data import SEED

BASE_NAMES = ["lr_word", "ridge_word", "ridge_emb", "knn_g", "knn_p", "auth_mean", "auth_max",
              "auth_cnt", "venyr_prior", "lr_char", "ridge_word_emb", "clus12_te", "clus40_te",
              "assoc_max", "assoc_min", "assoc_mean", "gbdt_svd", "ridge_emb_vi", "lr_title_word", "ridge_char", "ord_lr", "mlp_emb", "lr_abs_char"]


def _gain(y):
    return (np.asarray(y, float) - 1) / 4.0


def base_fit_predict(S, tr, ytr, va):
    """Return (len(va), n_base) matrix of base predictions, fitted on `tr` only."""
    g = _gain(ytr); pos = (ytr > 1).astype(int)
    out = []
    m = LogisticRegression(C=2.0, max_iter=300).fit(S.Xw_v[tr], pos); out.append(m.predict_proba(S.Xw_v[va])[:, 1])
    m = Ridge(alpha=3.0).fit(S.Xw_v[tr], g); out.append(m.predict(S.Xw_v[va]))
    m = Ridge(alpha=5.0).fit(S.E_v[tr], g); out.append(m.predict(S.E_v[va]))
    # kNN on embeddings (cosine; embeddings are L2-normalised)
    sim = S.E[va] @ S.E[tr].T
    k = 20
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    w = np.take_along_axis(sim, idx, 1).clip(0) ** 3 + 1e-9
    out.append((w * g[idx]).sum(1) / w.sum(1)); out.append((w * pos[idx]).sum(1) / w.sum(1))
    # author target encoding (smoothed)
    gm = g.mean(); m_s = 3.0
    sums, cnts = {}, {}
    for i, gi in zip(tr, g):
        for a in S.authors[i]:
            sums[a] = sums.get(a, 0.0) + gi; cnts[a] = cnts.get(a, 0) + 1
    am, ax, ac = [], [], []
    for i in va:
        vals = [((sums[a] + m_s * gm) / (cnts[a] + m_s)) for a in S.authors[i] if a in sums]
        cs = [cnts[a] for a in S.authors[i] if a in sums]
        am.append(np.mean(vals) if vals else gm); ax.append(np.max(vals) if vals else gm); ac.append(np.sum(cs) if cs else 0)
    out += [np.array(am), np.array(ax), np.log1p(np.array(ac))]
    # venue x year prior
    d_s, d_c = {}, {}
    for i, gi in zip(tr, g):
        d_s[S.vy_code[i]] = d_s.get(S.vy_code[i], 0) + gi; d_c[S.vy_code[i]] = d_c.get(S.vy_code[i], 0) + 1
    out.append(np.array([(d_s.get(S.vy_code[i], 0) + 5 * gm) / (d_c.get(S.vy_code[i], 0) + 5) for i in va]))
    m = LogisticRegression(C=1.0, max_iter=300).fit(S.Xc_v[tr], pos); out.append(m.predict_proba(S.Xc_v[va])[:, 1])
    m = Ridge(alpha=2.0).fit(sparse.hstack([S.Xw_v[tr], sparse.csr_matrix(S.E[tr])]).tocsr(), g)
    out.append(m.predict(sparse.hstack([S.Xw_v[va], sparse.csr_matrix(S.E[va])]).tocsr()))
    # clustering: smoothed target encoding of k-means clusters (k=12, 40) fitted on all text, labels from `tr` only
    for km in (S.km12, S.km40):
        K = km.max() + 1
        cs = np.bincount(km[tr], weights=g, minlength=K); cc = np.bincount(km[tr], minlength=K)
        te_ = (cs + 5 * gm) / (cc + 5); out.append(te_[km[va]])
    # itemset / association rules: single items and pairs, consequent = "high" (y>=3)
    hi = (ytr >= 3).astype(np.float32); p0 = hi.mean()
    It = S.items[tr]; sup = np.asarray(It.sum(0)).ravel(); posc = np.asarray(It.T @ hi).ravel()
    ok = np.where(sup >= 10)[0]
    conf1 = (posc[ok] + 5 * p0) / (sup[ok] + 5)
    sc_mat = np.zeros(S.items.shape[1]); sc_mat[ok] = np.log(conf1 / p0)
    # pairs among the 400 most supported items
    top = ok[np.argsort(-sup[ok])[:400]]
    It_t = It[:, top].tocsc(); co = (It_t.T @ It_t).toarray(); copos = np.asarray((It_t.multiply(hi[:, None])).T @ It_t.toarray()) if False else (It_t.multiply(hi[:, None]).T @ It_t).toarray()
    pair_score = np.where(co >= 10, np.log(((copos + 5 * p0) / (co + 5)) / p0), 0.0); np.fill_diagonal(pair_score, 0)
    Iv = S.items[va]
    am1, an1, aav = [], [], []
    Iv_t = Iv[:, top].toarray()
    Iv_all = Iv.tocsr()
    for r_ in range(len(va)):
        row_items = Iv_all[r_].indices
        sc = list(sc_mat[row_items][sc_mat[row_items] != 0])
        pr = np.where(Iv_t[r_] > 0)[0]
        if len(pr) > 1:
            sub = pair_score[np.ix_(pr, pr)][np.triu_indices(len(pr), 1)]
            sc += list(sub[sub != 0])
        if sc:
            sc = np.array(sc); am1.append(np.sort(sc)[-3:].mean()); an1.append(np.sort(sc)[:3].mean()); aav.append(sc.mean())
        else:
            am1.append(0.0); an1.append(0.0); aav.append(0.0)
    out += [np.array(am1), np.array(an1), np.array(aav)]
    # GBDT on SVD topics + venue/year (classification-style regression of gain)
    Xg_tr = np.column_stack([S.svd[tr], S.vcodes[tr], S.year[tr]]); Xg_va = np.column_stack([S.svd[va], S.vcodes[va], S.year[va]])
    gb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.04, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, verbose=-1, n_jobs=2, random_state=SEED).fit(Xg_tr, g)
    out.append(gb.predict(Xg_va))
    m = Ridge(alpha=8.0).fit(S.E_vi[tr], g); out.append(m.predict(S.E_vi[va]))
    m = LogisticRegression(C=2.0, max_iter=300).fit(S.Xt_v[tr], pos); out.append(m.predict_proba(S.Xt_v[va])[:, 1])
    m = Ridge(alpha=2.0).fit(S.Xc_v[tr], g); out.append(m.predict(S.Xc_v[va]))
    # ordinal: P(y>=k), k=2..5, expected gain = mean of the four cumulative probabilities
    ords = []
    for k in (2, 3, 4, 5):
        ords.append(LogisticRegression(C=1.5, max_iter=300).fit(S.Xwc[tr], (ytr >= k).astype(int)).predict_proba(S.Xwc[va])[:, 1])
    out.append(np.mean(ords, 0))
    from sklearn.neural_network import MLPRegressor
    mm = [MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=150, early_stopping=True, random_state=SEED + s_).fit(S.E_v[tr], g).predict(S.E_v[va]) for s_ in range(3)]
    out.append(np.mean(mm, 0))
    m = LogisticRegression(C=1.0, max_iter=300).fit(S.Xa_v[tr], pos); out.append(m.predict_proba(S.Xa_v[va])[:, 1])
    return np.column_stack(out)


def inner_oof(S, tr, ytr, n_splits=5, seed=SEED):
    B = np.zeros((len(tr), len(BASE_NAMES)))
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(tr, ytr):
        B[b] = base_fit_predict(S, tr[a], ytr[a], tr[b])
    return B


def meta_matrix(S, idx, B, cols=None):
    """Stacker input: selected base-learner outputs followed by metadata (venue code, year, #authors, has-abstract,
    title length (words/chars), short-paper tag, workshop/proceedings keyword)."""
    cols = list(range(B.shape[1])) if cols is None else list(cols)
    extra = np.column_stack([S.vcodes[idx], S.year[idx], S.nauth[idx], S.has_abs[idx], S.tl[idx], S.tc[idx], S.tag[idx], S.meta_kw[idx]])
    return np.hstack([B[:, cols], extra])


LGB_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=2)


def stack_parts(Mtr, ytr, Mva, nb, n_seeds=1, seed=SEED):
    """Level-2 models. Returns raw scores: ridge/LGBM expected-gain regressors and P(y>1) classifiers.
    The first `nb` columns are base-learner outputs, column `nb` is the venue code (categorical for LightGBM)."""
    g = (ytr - 1) / 4.0; pos = (ytr > 1).astype(int)
    mu, sd = Mtr[:, :nb].mean(0), Mtr[:, :nb].std(0) + 1e-9
    A, B = (Mtr[:, :nb] - mu) / sd, (Mva[:, :nb] - mu) / sd
    out = {"ridge_reg": Ridge(alpha=30).fit(A, g).predict(B),
           "ridge_clf": LogisticRegression(C=1 / 30, max_iter=500).fit(A, pos).predict_proba(B)[:, 1]}
    out["lgb_reg"] = np.mean([lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed + k).fit(Mtr, g, categorical_feature=[nb]).predict(Mva)
                              for k in range(n_seeds)], 0)
    out["lgb_clf"] = np.mean([lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed + k).fit(Mtr, pos, categorical_feature=[nb]).predict_proba(Mva)[:, 1]
                              for k in range(n_seeds)], 0)
    return out


def rank01(v):
    return rankdata(v) / len(v)


def blend(parts):
    """Priority utility = rank-average of expected gain (weight 1) and P(y>1) (weight 0.5) from ridge and LightGBM."""
    return rank01(parts["ridge_reg"]) + rank01(parts["lgb_reg"]) + 0.5 * rank01(parts["ridge_clf"]) + 0.5 * rank01(parts["lgb_clf"])
