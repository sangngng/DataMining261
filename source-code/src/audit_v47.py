"""
audit_v47.py:
Comprehensive integrity and validation audit for submission_v47_grand_apex_pentad_374.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

def run_audit():
    sub_path = SUB_DIR / "submission_v47_grand_apex_pentad_374.csv"
    test_path = DATA_DIR / "test_with_abstracts.csv"

    assert sub_path.exists(), f"File not found: {sub_path}"
    sub = pd.read_csv(sub_path)
    test = pd.read_csv(test_path)

    print("=== V47 AUDIT: STRUCTURAL INTEGRITY ===")
    assert len(sub) == 596, f"Length must be 596, got {len(sub)}"
    assert list(sub.columns) == ['id', 'priority'], f"Invalid columns: {list(sub.columns)}"
    assert (sub['id'] == test['id']).all(), "ID sequence mismatch!"
    print(" [x] Length: 596 rows verified")
    print(" [x] Column schema: ['id', 'priority'] verified")
    print(" [x] ID alignment: 100% matched to test set")

    print("\n=== V47 AUDIT: 374 BUDGET LAW & ZERO TIES ===")
    active_count = (sub['priority'] > 0).sum()
    pruned_count = (sub['priority'] == 0).sum()
    unique_count = len(sub['priority'].unique())
    print(f" [x] Active papers: {active_count} (Expected: 374)")
    print(f" [x] Pruned papers: {pruned_count} (Expected: 222)")
    print(f" [x] Unique priorities: {unique_count} (Expected: 375, Zero Ties Guaranteed!)")
    assert active_count == 374
    assert pruned_count == 222
    assert unique_count == 375

    print("\n=== V47 AUDIT: CONVEXITY & MONOTONICITY ===")
    active_sub = sub[sub['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
    p_vals = active_sub['priority'].values
    assert (np.diff(p_vals) < 0).all(), "Priority values must be strictly strictly monotonically decreasing!"
    assert p_vals[0] > 1.20, f"Peak priority should be > 1.20, got {p_vals[0]}"
    assert p_vals[-1] > 0.05, f"Lowest active priority should be > 0.05, got {p_vals[-1]}"
    print(f" [x] Priority range: [{p_vals[-1]:.6f}, {p_vals[0]:.6f}]")
    print(" [x] Monotonicity: 100% strictly decreasing")

    print("\n=== V47 AUDIT: BENCHMARK COMPARISONS (VS V45 CHAMPION) ===")
    v45 = pd.read_csv(SUB_DIR / "submission_v45_apex_triad_consensus_374.csv")
    v45_act = v45[v45['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

    order_v47 = list(active_sub['id'].values)
    order_v45 = list(v45_act['id'].values)

    # Check Top 25
    top25_match = (order_v47[:25] == order_v45[:25])
    print(f" [x] Top 25 matches V45: {top25_match}")
    assert top25_match, "Top 25 must be identical to V45!"

    # Check Saribatur #878
    r_878 = order_v47.index(878) + 1
    print(f" [x] Saribatur #878 rank: {r_878} (Expected: 22, Private Zenith Locked!)")
    assert r_878 == 22, f"Expected 878 at rank 22, got {r_878}"

    # Check shifts
    shifts = [abs(order_v47.index(pid) - order_v45.index(pid)) for pid in order_v47]
    num_shifted = sum(1 for s in shifts if s > 0)
    max_shift = max(shifts)
    print(f" [x] Number of shifted papers: {num_shifted} (Expected: 8)")
    print(f" [x] Maximum shift: {max_shift} (Expected: 1)")
    assert num_shifted == 8
    assert max_shift == 1

    print("\n=== V47 SHIFT AUDIT DETAILS ===")
    for pid in order_v47:
        r_old = order_v45.index(pid) + 1
        r_new = order_v47.index(pid) + 1
        if r_old != r_new:
            title = str(test.loc[test['id']==pid, 'title'].values[0])
            print(f" ID {pid:4d}: Rank {r_old:2d} -> {r_new:2d} ({r_new - r_old:+d}) | {title[:50]}")

    corr_v45 = spearmanr(sub['priority'], v45['priority']).statistic
    print(f"\n [x] Spearman correlation with V45: {corr_v45:.6f}")
    assert corr_v45 > 0.9999, f"Correlation too low: {corr_v45}"

    print("\n>>> ALL V47 INTEGRITY AUDITS PASSED 100%! READY FOR KAGGLE SUBMISSION <<<")

if __name__ == '__main__':
    run_audit()
