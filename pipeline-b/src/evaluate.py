"""Local validation: OOF utilities (reps x n) -> score on simulated 298-paper splits whose venue x year mix matches
the test set (test has no ICLP and more 2016-2021 papers). Same splits for every method; reports Q, R, Score."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from metric import score
tr = pd.read_csv("work/train.csv"); te = pd.read_csv("work/test.csv"); y = tr.Label.values
key = lambda d: (d.venue + "_" + d.year.clip(2016, 2025).astype(str)).values
ktr, kte = key(tr), key(te)
QUOTA = {k: int(round(298 * c / len(te))) for k, c in pd.Series(kte).value_counts().items()}
POOLS = {k: np.where(ktr == k)[0] for k in QUOTA}
r01 = lambda v: rankdata(v) / len(v)
def prio(u, keep):
    n = len(u); m = int(round(keep * n)); o = np.argsort(-u, kind="stable"); p = np.zeros(n); p[o[:m]] = np.linspace(10, 0.01, m); return p
def evaluate(cands, keep=0.65, n_sims=2000, seed=0, ref=None):
    rng = np.random.default_rng(seed); reps = next(iter(cands.values())).shape[0]
    res = {k: [] for k in cands}
    for _ in range(n_sims):
        idx = np.concatenate([rng.choice(POOLS[k], min(q, len(POOLS[k])), replace=False) for k, q in QUOTA.items()]); r = rng.integers(reps)
        for k, U in cands.items(): res[k].append(score(tr.id.values[idx], prio(U[r][idx], keep), y[idx], True))
    ref = ref or next(iter(cands)); b = np.array(res[ref])[:, 0]; rows = []
    for k, v in res.items():
        v = np.array(v); d = v[:, 0] - b
        rows.append({"method": k, "AUC": np.mean([roc_auc_score(y > 1, cands[k][r]) for r in range(reps)]), "Q": v[:, 1].mean(), "R": v[:, 2].mean(),
                     "Score": v[:, 0].mean(), "sd": v[:, 0].std(), "diff": d.mean(), "P>0": (d > 0).mean()})
    return pd.DataFrame(rows).round(4)
