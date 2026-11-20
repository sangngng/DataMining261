"""Taught-approach learners on sparse metadata (no pretrained models):
 - classification: logistic P(y>1); ordinal: mean of logistic P(y>=k), k=2..5
 - regression: ridge on gain (y-1)/4
 - itemset rules: frequent 1-/2-itemsets of {title words, authors, venue, year-bin}; per paper, mean smoothed
   confidence of 'worthwhile' and mean gain over its 5 most supported matched itemsets (association rules X -> y>1).
Usage: python src/models_local.py RUN  (RUN = b or c, selects work/folds_RUN.csv)
Output: work/local_oof_RUN.npz (3 reps x n) and work/local_test_RUN.npz."""
import re, sys, itertools, collections, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
tr = pd.read_csv("work/train.csv"); te = pd.read_csv("work/test.csv"); RUN = sys.argv[1] if len(sys.argv) > 1 else "b"; F = pd.read_csv(f"work/folds_{RUN}.csv"); y = tr.Label.values
ALL = pd.concat([tr, te], ignore_index=True); n = len(tr)
r01 = lambda v: rankdata(v) / len(v)
def authors(s): return [a.strip() for a in str(s).split(",") if a.strip()] if isinstance(s, str) else []
def cat(d): return pd.DataFrame({"v": d.venue, "yb": d.year.clip(2016, 2025).astype(str), "vy": d.venue + (d.year >= 2021).astype(str),
                                 "pro": d.title.str.contains(r"Proceedings|Conference|Volume|Part [IVX]", regex=True).astype(str)})
def sparse(fit, ds):
    tw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(fit.title)
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=40000).fit(fit.title)
    au = CountVectorizer(tokenizer=authors, lowercase=False, min_df=2, binary=True, token_pattern=None).fit(fit.authors.fillna(""))
    oh = OneHotEncoder(handle_unknown="ignore").fit(cat(fit))
    return [sp.hstack([tw.transform(d.title), 0.6 * tc.transform(d.title), 0.8 * au.transform(d.authors.fillna("")), oh.transform(cat(d)),
                       sp.csr_matrix(np.c_[np.log1p(d.authors.map(lambda s: len(authors(s)))) / 3, (d.year.values - 2020) / 5])]).tocsr() for d in ds]
def items(r):
    w = {"w:" + t for t in re.findall(r"[a-z][a-z\-]{2,}", r.title.lower()) if t not in ENGLISH_STOP_WORDS}
    return w | {"a:" + a for a in authors(r.authors)} | {"v:" + r.venue, "y:" + str(min(max(r.year, 2016), 2025) // 3)}
IT = [items(r) for r in ALL.itertuples()]
def rules_fit(idx, ya, min_sup=8):
    cnt = collections.Counter(); pos = collections.Counter(); gain = collections.Counter()
    for i, l in zip(idx, ya):
        s = sorted(IT[i]); keys = s + [a + "|" + b for a, b in itertools.combinations([x for x in s if not x.startswith("y:")], 2)]
        for k in keys: cnt[k] += 1; pos[k] += l > 1; gain[k] += (l - 1) / 4
    p0, g0, al = np.mean(ya > 1), np.mean((ya - 1) / 4), 5.0
    return {k: (c, (pos[k] + al * p0) / (c + al), (gain[k] + al * g0) / (c + al)) for k, c in cnt.items() if c >= min_sup}, (p0, g0)
def rules_predict(R, prior, idx, top=5):
    out = np.zeros((len(idx), 2))
    for j, i in enumerate(idx):
        s = sorted(IT[i]); keys = s + [a + "|" + b for a, b in itertools.combinations([x for x in s if not x.startswith("y:")], 2)]
        m = sorted([R[k] for k in keys if k in R], key=lambda t: -t[0])[:top]
        out[j] = (np.mean([t[1] for t in m]), np.mean([t[2] for t in m])) if m else prior
    return out
def learners(a, b, Xa, Xb):
    ya = y[a]
    lr = LogisticRegression(C=2.0, max_iter=3000).fit(Xa, ya > 1).predict_proba(Xb)[:, 1]
    rg = Ridge(alpha=3.0).fit(Xa, (ya - 1) / 4).predict(Xb)
    od = np.mean([LogisticRegression(C=2.0, max_iter=3000).fit(Xa, ya >= k).predict_proba(Xb)[:, 1] for k in (2, 3, 4, 5)], 0)
    R, pr = rules_fit(a, ya); ru = rules_predict(R, pr, b)
    return {"lr": lr, "ridge": rg, "ord": od, "rule_pos": ru[:, 0], "rule_gain": ru[:, 1]}
oof = {}
for r in range(3):
    f = F[f"fold{r}"].values
    for k in range(5):
        a, b = np.where(f != k)[0], np.where(f == k)[0]
        Xa, Xb = sparse(tr.iloc[a], [tr.iloc[a], tr.iloc[b]])
        for nm, v in learners(a, b, Xa, Xb).items(): oof.setdefault(nm, np.zeros((3, n)))[r, b] = v
    print("rep", r, flush=True)
Xa, Xb = sparse(tr, [tr, te]); tst = learners(np.arange(n), np.arange(n, n + len(te)), Xa, Xb)
np.savez(f"work/local_oof_{RUN}.npz", **oof); np.savez(f"work/local_test_{RUN}.npz", **tst); print("saved")
