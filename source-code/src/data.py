"""Data loading, near-duplicate groups and repeated stratified-group folds."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42


def load():
    tr = pd.read_csv(DATA / "train_with_abstracts.csv")
    te = pd.read_csv(DATA / "test_with_abstracts.csv")
    z = np.load(DATA / "clusters_k12.npz")
    return tr, te, z["tr_embs"], z["te_embs"]


def text_of(df):
    ab = df["abstract"].fillna("") if "abstract" in df else ""
    return (df["title"].fillna("") + ". " + ab).str.strip()


def str_rank(ids):
    s = np.array([str(i) for i in ids])
    return np.argsort(np.argsort(s, kind="stable"), kind="stable")


# ---------------------------------------------------------------- groups & folds
def near_dup_groups(texts, thr=0.8):
    from scipy.sparse.csgraph import connected_components
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
    S = (X @ X.T).tocoo()
    keep = (S.data >= thr) & (S.row < S.col)
    A = sparse.coo_matrix((np.ones(keep.sum()), (S.row[keep], S.col[keep])), shape=S.shape)
    n, lab = connected_components(A, directed=False)
    return lab


def repeated_folds(y, groups, n_splits=5, n_repeats=3, seed=SEED):
    """Repeated stratified *group* K-fold (groups = near duplicate clusters)."""
    from sklearn.model_selection import StratifiedGroupKFold
    out = []
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (a, b) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
            out.append((r, k, a, b))
    return out
