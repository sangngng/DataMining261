"""Analysis: CV ceiling. Perfect ranking (true labels, keep exactly the worthwhile papers) and the final utility mixed with
the true ranking, to show how much AUC a given Score gain requires."""
import sys, numpy as np
sys.path.insert(0, "src")
import evaluate as E
from metric import score
from pipelines import oof, r01
from sklearn.metrics import roc_auc_score
A, B, C = (oof(k) for k in "ABC"); U = (A + B + C) / 3; y = E.y
rng0 = np.random.default_rng(0)
def sc(Ux, keep=0.65, N=2000, seed=5):
    rng = np.random.default_rng(seed); out = []
    for _ in range(N):
        idx = np.concatenate([rng.choice(E.POOLS[k], min(q, len(E.POOLS[k])), replace=False) for k, q in E.QUOTA.items()]); r = rng.integers(Ux.shape[0])
        out.append(score(E.tr.id.values[idx], E.prio(Ux[r][idx], keep), y[idx]))
    return np.mean(out), np.std(out)
oracle = np.stack([y + 1e-3 * rng0.random(len(y))] * 2)
print("oracle (true labels), keep only y>1:", [round(v, 4) for v in sc(oracle, keep=float(np.mean(y > 1)))])
for noise in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    X = np.stack([r01(U[r] + noise * r01(y + 1e-3 * rng0.random(len(y)))) for r in range(2)])
    print(f"final + {noise} x true-rank mix: AUC {np.mean([roc_auc_score(y>1, X[r]) for r in range(2)]):.3f}  Score {sc(X)[0]:.4f}")
