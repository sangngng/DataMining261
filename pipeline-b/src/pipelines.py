"""Rank utilities of the independently trained pipelines (out-of-fold for CV, and for the test set).
  A  = source-code/ (blend_ft4): 0.5 * mean(stack, MiniLM-title, BGE-title) + 0.5 * mean of 4 fine-tuned encoders
  B  = this folder, folds_b: 0.4 * mean(sparse lr/ridge/ordinal, MiniLM-emb learner, BGE-emb learner) + 0.6 * mean of 4 fine-tuned encoders
  C  = same code as B, folds_c and other seeds
Every utility is rank-normalised to (0, 1] so that pipelines can be averaged with equal weights."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
r01 = lambda v: rankdata(v) / len(v)
R = lambda a: np.stack([r01(x) for x in a])
TAGS = ("bge", "e5", "sci", "mpnet")
A_DIR = "../source-code/data/"


def _base(L, M, rank):
    return ((rank(L["lr"]) + rank(L["ridge"]) + rank(L["ord"])) / 3 + (rank(M["minilm_reg"]) + rank(M["minilm_clf"])) / 2
            + (rank(M["bge_reg"]) + rank(M["bge_clf"])) / 2) / 3


def oof(name, n_reps=2, w_base=0.4):
    """(n_reps x 2494) OOF utilities. n_reps=2 because pipeline A has two repeats for every fine-tuned encoder."""
    if name == "A":
        P = pd.read_csv(A_DIR + "oof_blend_a.csv")
        ft = {t: np.load(A_DIR + f"ft_{t}_oof{'5' if t in ('sci', 'mpnet') else ''}.npy") for t in TAGS}
        return R(np.stack([0.5 * P[f"u{r}"].values + 0.5 * sum(r01(ft[t][r]) for t in TAGS) / 4 for r in range(n_reps)]))
    run, ft_run = {"B": ("b", "b"), "C": ("c", "c")}[name]
    L, M = dict(np.load(f"work/local_oof_{run}.npz")), dict(np.load(f"work/emb_oof_{run}.npz"))
    FT = sum(R(np.load(f"work/ft_{ft_run}_{t}_oof.npy")) for t in TAGS) / 4
    return R((w_base * _base(L, M, R) + (1 - w_base) * FT)[:n_reps])


def test(name, ranked=True):
    """(596,) test utility in the row order of work/test.csv."""
    if name == "A":                     # cd source-code; python -m src.make_submission --model blend_ft4 --dump-utility ../pipeline-b/work/utility_a_test.csv
        te = pd.read_csv("work/test.csv")
        return r01(pd.read_csv("work/utility_a_test.csv").set_index("id").u.reindex(te.id).values)
    run, ft_run = {"B": ("b", "b"), "C": ("c", "c")}[name]
    L, M = dict(np.load(f"work/local_test_{run}.npz")), dict(np.load(f"work/emb_test_{run}.npz"))
    u = 0.4 * _base(L, M, r01) + 0.6 * sum(r01(np.load(f"work/ft_{ft_run}_{t}_test.npy")) for t in TAGS) / 4
    return r01(u) if ranked else u
