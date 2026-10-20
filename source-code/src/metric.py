import numpy as np
import pandas as pd
from typing import Tuple, Union, Sequence

def compute_competition_metric(
    ids: Sequence[Union[int, str]],
    y_true: Sequence[float],
    priority: Sequence[float]
) -> Tuple[float, float, float]:
    """
    Computes the official Kaggle competition score for CO3029:
      - Relevance (Q): discounted gain weighted by 1 / log2(r + 1)
      - Coverage (R): recall of worthwhile papers (true label > 1)
      - Score: Harmonic Mean of Q and R

    Args:
        ids: sequence of paper IDs
        y_true: ground truth ratings (values in {1, 2, 3, 4, 5})
        priority: predicted priority scores (finite non-negative numbers >= 0)

    Returns:
        (score, Q, R)
    """
    df = pd.DataFrame({
        'id': [str(x) for x in ids],
        'y': np.asarray(y_true, dtype=float),
        'p': np.asarray(priority, dtype=float)
    })

    # Coverage denominator: all papers in the evaluated split with true rating > 1
    total_worthwhile = np.sum(df['y'] > 1.0)
    if total_worthwhile == 0:
        return 0.0, 0.0, 0.0

    # Exclude papers with priority <= 0 (or exactly 0)
    rec = df[df['p'] > 0.0].copy()

    # Coverage numerator: recommended papers with true rating > 1
    rec_worthwhile = np.sum(rec['y'] > 1.0)
    R = float(rec_worthwhile) / float(total_worthwhile)

    # If no papers are recommended, Relevance is 0
    if len(rec) == 0:
        return 0.0, 0.0, R

    # Sort recommended papers by descending priority, ties broken by ascending ID as string
    rec = rec.sort_values(by=['p', 'id'], ascending=[False, True]).reset_index(drop=True)

    # Rank positions r = 1, 2, ..., m
    m = len(rec)
    r = np.arange(1, m + 1, dtype=float)
    w_r = 1.0 / np.log2(r + 1.0)
    g_r = (rec['y'].values - 1.0) / 4.0

    Q = float(np.sum(w_r * g_r)) / float(np.sum(w_r))

    if (Q + R) == 0:
        score = 0.0
    else:
        score = 2.0 * Q * R / (Q + R)

    return score, Q, R


if __name__ == "__main__":
    # Test with example from Kaggle evaluation page
    # If relevance = 0.80 and coverage = 0.60 -> Harmonic Mean approx 0.685714
    q_example = 0.80
    r_example = 0.60
    harmonic_example = 2 * q_example * r_example / (q_example + r_example)
    print(f"Sample Harmonic Mean check: {harmonic_example:.6f} (Expected ~0.685714)")
