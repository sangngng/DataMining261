"""Choose the 3 final Kaggle submissions on CV. Kaggle scores a team by the BEST of its selected submissions on the
Private split, so we pick the triple with the highest expected max-score over simulated 298-paper splits
(same venue x year sampler as evaluate.py; every candidate is scored on the same splits).
    python src/select_final.py   -> reports/final_selection.txt"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "src")
import evaluate as E
from metric import score
from pipelines import oof
A, B, C = (oof(k) for k in "ABC")
U = {"A": A, "B": B, "AB": (A + B) / 2, "ABC": (A + B + C) / 3}
cands = {f"{u}@{k:.2f}": (u, k) for u in U for k in (0.55, 0.60, 0.65, 0.70, 0.75)}
rng = np.random.default_rng(2024); N = 3000; names = list(cands); S = np.zeros((N, len(names)))
for s in range(N):
    idx = np.concatenate([rng.choice(E.POOLS[k], min(q, len(E.POOLS[k])), replace=False) for k, q in E.QUOTA.items()]); r = rng.integers(A.shape[0])
    for j, n in enumerate(names):
        u, k = cands[n]; S[s, j] = score(E.tr.id.values[idx], E.prio(U[u][r][idx], k), E.y[idx])
single = pd.Series(S.mean(0), names).sort_values(ascending=False)
best = sorted(((S[:, list(t)].max(1).mean(), [names[i] for i in t]) for t in itertools.combinations(range(len(names)), 3)), reverse=True)[:10]
out = ["== single candidates (mean Score) ==", single.round(4).to_string(), "", "== best triples (mean of max Score over the 3) =="]
out += [f"{m:.4f}  {t}" for m, t in best]
for fixed in (["ABC@0.65", "AB@0.65", "A@0.65"], ["ABC@0.65", "AB@0.65", "B@0.60"]):
    out.append(f"reference {fixed}: {S[:, [names.index(n) for n in fixed]].max(1).mean():.4f}")
txt = "\n".join(out); print(txt); open("reports/final_selection.txt", "w").write(txt)
