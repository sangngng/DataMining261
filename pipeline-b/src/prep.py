"""Step 1. Read the raw Kaggle files (data/), repair the unquoted Test.csv, write clean copies to work/, and build the CV
folds. Folds: StratifiedGroupKFold, 5 folds x 3 repetitions; groups join exact-DOI duplicates and near-duplicate titles
(TF-IDF cosine >= 0.8), so related papers never sit on both sides of a split.
    run b -> work/folds_b.csv (seeds 2026, 2027, 2028)
Usage (from pipeline-b/):  python src/prep.py"""
import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
sys.path.insert(0, "src")
from io_data import read_train, read_test

RUNS = {"b": 2026}
tr = read_train("data/train.csv"); te = read_test("data/Test.csv"); ss = pd.read_csv("data/Submission.csv")
assert set(te.id) == set(ss.id) and len(te) == 596 and tr.id.is_unique
tr.to_csv("work/train.csv", index=False); te.to_csv("work/test.csv", index=False); ss.to_csv("work/sample_submission.csv", index=False)
X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(tr.title.str.lower())
S = (X @ X.T).tocoo(); A = sp.coo_matrix((np.ones((S.data >= 0.8).sum()), (S.row[S.data >= 0.8], S.col[S.data >= 0.8])), shape=S.shape)
d = tr.doi.str.lower(); same = d.duplicated(keep=False)
rows = []
for _, g in tr[same].groupby(d[same]): idx = g.index.values; rows += [(idx[0], j) for j in idx[1:]]
if rows: A = A + sp.coo_matrix((np.ones(len(rows)), tuple(np.array(rows).T)), shape=S.shape)
ng, grp = connected_components(A, directed=False)
print("rows", len(tr), "groups", ng, "multi-paper groups", (np.bincount(grp) > 1).sum(), "dup DOIs", same.sum())
for run, seed0 in RUNS.items():
    F = pd.DataFrame({"id": tr.id, "group": grp})
    for r in range(3):
        f = np.zeros(len(tr), int)
        for k, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed0 + r).split(tr, tr.Label, grp)): f[b] = k
        F[f"fold{r}"] = f
    F.to_csv(f"work/folds_{run}.csv", index=False); print("run", run, F.groupby("fold0").size().to_dict())
