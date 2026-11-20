"""Embedding learners (frozen pretrained title vectors: all-MiniLM-L6-v2, bge-base-en-v1.5).
Per encoder, features computed from the training fold only:
  * kNN label statistics (k=5/20, similarity-weighted gain), nearest liked (y>=4) / disliked (y=1) similarity;
  * clustering: KMeans (k=20) fitted on train+test vectors (unsupervised, declared), out-of-fold cluster mean gain
    and P(y>1), distance to own centroid;
  * 32-d SVD of the vectors, venue code, year, #authors, proceedings flag, out-of-fold author target encoding;
  -> LightGBM regressor (gain) + classifier (P(y>1)), 3 seeds.
Usage: python src/models_emb.py RUN  (RUN = b or c).  Output: work/emb_oof_RUN.npz, work/emb_test_RUN.npz"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
tr = pd.read_csv("work/train.csv"); te = pd.read_csv("work/test.csv"); RUN = sys.argv[1] if len(sys.argv) > 1 else "b"; F = pd.read_csv(f"work/folds_{RUN}.csv"); y = tr.Label.values; n = len(tr)
ALL = pd.concat([tr, te], ignore_index=True); r01 = lambda v: rankdata(v) / len(v)
P = dict(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=25, subsample=0.85, subsample_freq=1, colsample_bytree=0.75, verbose=-1, n_jobs=4)
AU = [[a.strip() for a in str(s).split(",") if a.strip()] if isinstance(s, str) else [] for s in ALL.authors]
vcode = ALL.venue.astype("category").cat.codes.values; pro = ALL.title.str.contains(r"Proceedings|Conference|Volume|Part [IVX]", regex=True).values
nau = np.array([len(a) for a in AU]); yr = ALL.year.values
def author_te(a_idx, rows, self_ex):
    g0 = np.mean((y[a_idx] - 1) / 4); s = {}; c = {}
    for i in a_idx:
        for au in AU[i]: s[au] = s.get(au, 0) + (y[i] - 1) / 4; c[au] = c.get(au, 0) + 1
    out = np.zeros((len(rows), 2))
    for j, i in enumerate(rows):
        v = []
        for au in AU[i]:
            cc, ss = c.get(au, 0), s.get(au, 0)
            if self_ex: cc, ss = cc - 1, ss - (y[i] - 1) / 4
            if cc > 0: v.append(((ss + 3 * g0) / (cc + 3), cc))
        out[j] = (max(t[0] for t in v), sum(t[1] for t in v)) if v else (g0, 0)
    return out
def feats(E, Z, lab, a, rows, self_ex):
    S = E[rows] @ E[a].T
    if self_ex: S[np.arange(len(rows)), np.arange(len(rows))] = -9
    o = np.argsort(-S, 1)[:, :20]; L = y[a][o]; Ss = np.take_along_axis(S, o, 1); w = np.maximum(Ss, 0) ** 3
    pos, neg = y[a] >= 4, y[a] == 1
    dp = np.where(pos[None], S, -9).max(1); dn = np.where(neg[None], S, -9).max(1)
    cg = pd.Series((y[a] - 1) / 4).groupby(lab[a]).agg(["sum", "count"]); cp = pd.Series(y[a] > 1).groupby(lab[a]).sum()
    g0 = np.mean((y[a] - 1) / 4); p0 = np.mean(y[a] > 1); cl = lab[rows]
    sm, ct = cg["sum"].reindex(cl).fillna(0).values, cg["count"].reindex(cl).fillna(0).values; spos = cp.reindex(cl).fillna(0).values
    if self_ex: sm, ct, spos = sm - (y[rows] - 1) / 4, ct - 1, spos - (y[rows] > 1)
    return np.column_stack([(L[:, :5] > 1).mean(1), (L > 1).mean(1), ((L - 1) / 4 * w).sum(1) / (w.sum(1) + 1e-9), (L[:, :5] - 1).mean(1) / 4,
                            Ss[:, :5].mean(1), dp, dn, dp - dn, (sm + 10 * g0) / (ct + 10), (spos + 10 * p0) / (ct + 10)])
def run(E, Z, lab, dist, a, b):
    meta = lambda r: np.column_stack([Z[r], vcode[r], yr[r], nau[r], pro[r], dist[r]])
    Fa = np.hstack([feats(E, Z, lab, a, a, True), author_te(a, a, True), meta(a)]); Fb = np.hstack([feats(E, Z, lab, a, b, False), author_te(a, b, False), meta(b)])
    reg = np.mean([lgb.LGBMRegressor(**P, random_state=s).fit(Fa, (y[a] - 1) / 4).predict(Fb) for s in range(3)], 0)
    clf = np.mean([lgb.LGBMClassifier(**P, random_state=s).fit(Fa, y[a] > 1).predict_proba(Fb)[:, 1] for s in range(3)], 0)
    return reg, clf
oof, tst = {}, {}
for tag in ("minilm", "bge"):
    E = np.vstack([np.load(f"work/emb_{tag}_train.npy"), np.load(f"work/emb_{tag}_test.npy")]); E /= np.linalg.norm(E, axis=1, keepdims=True)
    Z = TruncatedSVD(32, random_state=0).fit_transform(E)
    km = KMeans(20, n_init=10, random_state=0).fit(E); lab = km.labels_; dist = np.linalg.norm(E - km.cluster_centers_[lab], axis=1)
    for r in range(3):
        f = F[f"fold{r}"].values
        for k in range(5):
            a, b = np.where(f != k)[0], np.where(f == k)[0]; reg, clf = run(E, Z, lab, dist, a, b)
            oof.setdefault(tag + "_reg", np.zeros((3, n)))[r, b] = reg; oof.setdefault(tag + "_clf", np.zeros((3, n)))[r, b] = clf
        print(tag, "rep", r, flush=True)
    reg, clf = run(E, Z, lab, dist, np.arange(n), np.arange(n, len(ALL))); tst[tag + "_reg"], tst[tag + "_clf"] = reg, clf
np.savez(f"work/emb_oof_{RUN}.npz", **oof); np.savez(f"work/emb_test_{RUN}.npz", **tst); print("saved")
