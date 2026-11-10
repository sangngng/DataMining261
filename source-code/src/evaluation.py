"""Fast re-implementation of the official metric (validated against src/metric.py in tests) and
venue-matched 298-paper split simulation."""
import numpy as np
from .data import str_rank

def metric_fast(y, p, id_str_rank):
    """Official metric. y in 1..5, p>=0, id_str_rank = rank of str(id) (tie-break asc).
    Returns (score, Q, R)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    tot = (y > 1).sum()
    if tot == 0:
        return 0.0, 0.0, 0.0
    m = p > 0
    if m.sum() == 0:
        return 0.0, 0.0, 0.0
    yy, pp, ii = y[m], p[m], id_str_rank[m]
    order = np.lexsort((ii, -pp))
    yy = yy[order]
    r = np.arange(1, len(yy) + 1)
    w = 1.0 / np.log2(r + 1)
    Q = (w * (yy - 1) / 4).sum() / w.sum()
    R = (yy > 1).sum() / tot
    if Q == 0 or R == 0:
        return 0.0, Q, R
    return 2 * Q * R / (Q + R), Q, R


def priority_from_utility(u, keep_frac):
    """Convert a utility vector to priority: top keep_frac kept (>0, strictly distinct, descending), rest 0."""
    n = len(u); m = int(round(keep_frac * n))
    order = np.argsort(-u, kind="stable")
    p = np.zeros(n)
    p[order[:m]] = np.linspace(10, 0.01, m)
    return p


class VenueMatchedSampler:
    """Random 298-paper subsets of out-of-fold predictions that mimic the test venue mix (test has no `iclp`)."""
    def __init__(self, tr, te, y, size=298):
        self.y = np.asarray(y); self.rk = str_rank(tr["id"].values)
        counts = te["venue"].value_counts()
        self.quota = {v: int(round(size * c / len(te))) for v, c in counts.items()}
        self.pools = {v: np.where(tr["venue"].values == v)[0] for v in self.quota}

    def draw(self, rng, odds=1.0):
        """odds > 1 re-weights sampling towards worthwhile papers (label-shift scenario)."""
        idx = []
        for v, q in self.quota.items():
            pool = self.pools[v]; w = np.where(self.y[pool] > 1, odds, 1.0); w = w / w.sum()
            idx.append(rng.choice(pool, q, replace=False, p=None if odds == 1.0 else w))
        return np.concatenate(idx)

    def curve(self, u_rep, fracs, n_sims=300, seed=0, odds=1.0):
        """Mean/std of (score, Q, R) per keep fraction. u_rep: (n_rep, N) utilities."""
        rng = np.random.default_rng(seed); res = {f: [] for f in fracs}
        for _ in range(n_sims):
            idx = self.draw(rng, odds); u = u_rep[rng.integers(len(u_rep))][idx]
            for f in fracs:
                res[f].append(metric_fast(self.y[idx], priority_from_utility(u, f), self.rk[idx]))
        return {f: (np.mean(v, 0), np.std(v, 0)) for f, v in res.items()}

    def paired(self, u1, u2, frac, n_sims=800, seed=7):
        """Paired score differences u1 - u2 on identical random splits."""
        rng = np.random.default_rng(seed); d = []
        for _ in range(n_sims):
            idx = self.draw(rng); r = rng.integers(len(u1))
            a = metric_fast(self.y[idx], priority_from_utility(u1[r][idx], frac), self.rk[idx])[0]
            b = metric_fast(self.y[idx], priority_from_utility(u2[r][idx], frac), self.rk[idx])[0]
            d.append(a - b)
        d = np.array(d)
        return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()
