"""Analysis: select papers and order them with different scores (selection drives R and the Q denominator, order drives Q only).
Result: no gain over one utility for both (differences <= 0.0007); classifier-only selection is worse."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import evaluate as E
from metric import score
from pipelines import oof, R, TAGS, _base
A, B, C = (oof(k) for k in "ABC"); U = (A + B + C) / 3
# classification-only and gain-only views for B and C
def views(run):
    L = dict(np.load(f"work/local_oof_{run}.npz")); M = dict(np.load(f"work/emb_oof_{run}.npz"))
    FT = sum(R(np.load(f"work/ft_{run}_{t}_oof.npy")) for t in TAGS) / 4
    clf = (R(L["lr"]) + R(L["ord"]) + R(M["minilm_clf"]) + R(M["bge_clf"])) / 4
    reg = (R(L["ridge"]) + R(M["minilm_reg"]) + R(M["bge_reg"])) / 3
    rule = (R(L["rule_pos"]) + R(L["rule_gain"])) / 2
    return R(clf[:2]), R(reg[:2]), R(FT[:2]), R(rule[:2])
cb, rb, fb, ub = views("b"); cc, rc, fc, uc = views("c")
CLF, REG, FT = R((cb + cc) / 2), R((rb + rc) / 2), R((fb + fc) / 2)
def run(sel, order, keep, N=3000, seed=5):
    rng = np.random.default_rng(seed); out = []
    for _ in range(N):
        idx = np.concatenate([rng.choice(E.POOLS[k], min(q, len(E.POOLS[k])), replace=False) for k, q in E.QUOTA.items()]); r = rng.integers(2)
        s, o = sel[r][idx], order[r][idx]; m = int(round(keep * len(idx)))
        chosen = np.argsort(-s, kind="stable")[:m]; p = np.zeros(len(idx))
        p[chosen[np.argsort(-o[chosen], kind="stable")]] = np.linspace(10, 0.01, m)
        out.append(score(E.tr.id.values[idx], p, E.y[idx], True))
    o = np.array(out); return o[:, 0].mean(), o[:, 1].mean(), o[:, 2].mean()
for k in (0.60, 0.65):
    print("keep", k)
    for nm, sel, order in [("final ABC/ABC", U, U), ("sel CLF+ABC, order ABC", R((CLF + U) / 2), U), ("sel ABC, order REG+ABC", U, R((REG + U) / 2)),
                           ("sel ABC, order FT+ABC", U, R((FT + U) / 2)), ("sel CLF+ABC, order FT+ABC", R((CLF + U) / 2), R((FT + U) / 2)), ("sel ABC, order FT", U, FT)]:
        print(f"  {nm:30s} Score {run(sel, order, k)[0]:.4f}")
