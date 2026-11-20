"""Official competition metric (Evaluation page): Score = 2QR/(Q+R) on one split."""
import numpy as np


def score(ids, priority, labels, return_parts=False):
    ids = np.asarray(ids).astype(str); p = np.asarray(priority, float); y = np.asarray(labels, float)
    m = p > 0
    order = np.lexsort((ids[m], -p[m]))                      # desc priority, ties by id as text
    ys = y[m][order]
    if len(ys) == 0:
        Q = 0.0
    else:
        w = 1 / np.log2(np.arange(1, len(ys) + 1) + 1)
        Q = float((w * (ys - 1) / 4).sum() / w.sum())
    tot = (y > 1).sum()
    R = float((ys > 1).sum() / tot) if tot > 0 else 0.0
    s = 0.0 if Q == 0 or R == 0 else 2 * Q * R / (Q + R)
    return (s, Q, R) if return_parts else s
