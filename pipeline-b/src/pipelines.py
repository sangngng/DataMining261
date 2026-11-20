"""Rank utilities of pipeline B (out-of-fold for CV, and for the test set).
  B  = this folder, folds_b: 0.4 * mean(sparse lr/ridge/ordinal, MiniLM-emb learner, BGE-emb learner) + 0.6 * mean of 4 fine-tuned encoders
Every utility is rank-normalised to (0, 1]."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
r01 = lambda v: rankdata(v) / len(v)
R = lambda a: np.stack([r01(x) for x in a])
TAGS = ("bge", "e5", "sci", "mpnet")


def _base(L, M, rank):
    return ((rank(L["lr"]) + rank(L["ridge"]) + rank(L["ord"])) / 3 + (rank(M["minilm_reg"]) + rank(M["minilm_clf"])) / 2
            + (rank(M["bge_reg"]) + rank(M["bge_clf"])) / 2) / 3


def oof(name, n_reps=2, w_base=0.4):
    """(n_reps x 2494) OOF utilities."""
    run, ft_run = {"B": ("b", "b")}[name]
    L, M = dict(np.load(f"work/local_oof_{run}.npz")), dict(np.load(f"work/emb_oof_{run}.npz"))
    FT = sum(R(np.load(f"work/ft_{ft_run}_{t}_oof.npy")) for t in TAGS) / 4
    return R((w_base * _base(L, M, R) + (1 - w_base) * FT)[:n_reps])


def test(name, ranked=True):
    """(596,) test utility in the row order of work/test.csv."""
    run, ft_run = {"B": ("b", "b")}[name]
    L, M = dict(np.load(f"work/local_test_{run}.npz")), dict(np.load(f"work/emb_test_{run}.npz"))
    u = 0.4 * _base(L, M, r01) + 0.6 * sum(r01(np.load(f"work/ft_{ft_run}_{t}_test.npy")) for t in TAGS) / 4
    return r01(u) if ranked else u
