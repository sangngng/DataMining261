"""Wider final check: 7 utilities (incl. base weight 0.2) x 8 keep fractions, 3 independent sets of 3,000 splits.
Result: best triple 0.6399 vs chosen triple 0.6394 -- within noise and chosen in-sample, so the selection was kept."""
import sys, itertools, numpy as np
sys.path.insert(0, "src")
import evaluate as E
from metric import score
from pipelines import oof
A, B, C = (oof(k) for k in "ABC"); B2, C2 = oof("B", w_base=0.2), oof("C", w_base=0.2)
U = {"A": A, "B": B, "C": C, "AB": (A + B) / 2, "ABC": (A + B + C) / 3, "ABCw2": (A + B2 + C2) / 3, "BC": (B + C) / 2}
KEEPS = (0.50, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70)
names = [(u, k) for u in U for k in KEEPS]
def sims(seed, N=3000):
    rng = np.random.default_rng(seed); S = np.zeros((N, len(names)))
    for s in range(N):
        idx = np.concatenate([rng.choice(E.POOLS[k], min(q, len(E.POOLS[k])), replace=False) for k, q in E.QUOTA.items()]); r = rng.integers(2)
        ids, yy = E.tr.id.values[idx], E.y[idx]
        for j, (u, k) in enumerate(names): S[s, j] = score(ids, E.prio(U[u][r][idx], k), yy)
    return S
Ss = [sims(sd) for sd in (101, 202, 303)]
single = np.mean([S.mean(0) for S in Ss], 0); o = np.argsort(-single)
print("top singles:", [(f"{names[i][0]}@{names[i][1]}", round(single[i], 4)) for i in o[:8]])
# triples: restrict search to top-30 singles for speed
top = o[:30]; res = []
for t in itertools.combinations(top, 3):
    res.append((np.mean([S[:, list(t)].max(1).mean() for S in Ss]), [np.round(S[:, list(t)].max(1).mean(), 4) for S in Ss], t))
res.sort(key=lambda x: -x[0])
nm = lambda t: [f"{names[i][0]}@{names[i][1]}" for i in t]
for m, per, t in res[:12]: print(round(m, 4), per, nm(t))
cur = [names.index(x) for x in (("ABC", 0.65), ("B", 0.60), ("A", 0.55))]
print("current pick", round(np.mean([S[:, cur].max(1).mean() for S in Ss]), 4), [np.round(S[:, cur].max(1).mean(), 4) for S in Ss])
