"""Analysis: which final pick wins in which kind of Private split. The Private split is an unknown random half of the
test set, so we simulate many test-like 298-paper splits and group them by the share of worthwhile papers (label > 1),
and also apply label shift (worthwhile papers made more / less likely by an odds factor), then report each pick's mean
Score and how often it is the best of the three. Uses CV out-of-fold predictions only (no leaderboard information)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import evaluate as E
from metric import score
from pipelines import oof
A, B, C = (oof(k) for k in "ABC")
PICKS = {"A+B+C keep .65": ((A + B + C) / 3, 0.65), "B keep .60": (B, 0.60), "A keep .55": (A, 0.55)}
y = E.y; good = y > 1
def draw(rng, odds):
    parts = []
    for k, q in E.QUOTA.items():
        P = E.POOLS[k]; w = np.where(good[P], odds, 1.0); w = w / w.sum()
        parts.append(rng.choice(P, min(q, len(P)), replace=False, p=w))
    return np.concatenate(parts)
rows = []
for odds in (0.6, 0.8, 1.0, 1.25, 1.6):
    rng = np.random.default_rng(int(odds * 100))
    for _ in range(2000):
        idx = draw(rng, odds); r = rng.integers(2); rec = {"odds": odds, "share": good[idx].mean()}
        for n, (U, k) in PICKS.items(): rec[n] = score(E.tr.id.values[idx], E.prio(U[r][idx], k), y[idx])
        rows.append(rec)
D = pd.DataFrame(rows); names = list(PICKS)
D["best"] = D[names].max(1); D["winner"] = D[names].idxmax(1)
D["share_bin"] = pd.cut(D.share, [0, .55, .60, .65, .70, 1], labels=["<55%", "55-60%", "60-65%", "65-70%", ">70%"])
pd.set_option("display.width", 200)
print("worthwhile share in the real train set: %.3f" % good.mean())
print("\n== by share of worthwhile papers in the split ==")
print(D.groupby("share_bin", observed=True).agg(n=("best", "size"), **{n: (n, "mean") for n in names}, best_of_3=("best", "mean")).round(4).to_string())
print("\n   how often each pick is the best of the three:")
print(pd.crosstab(D.share_bin, D.winner, normalize="index").round(2).to_string())
print("\n== by label-shift odds (1.0 = same as train) ==")
print(D.groupby("odds").agg(**{n: (n, "mean") for n in names}, best_of_3=("best", "mean")).round(4).to_string())
