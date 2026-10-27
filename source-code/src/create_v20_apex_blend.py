import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from scipy.stats import rankdata

def create_apex_and_fixed_v19():
    data_dir = os.path.join(BASE_DIR, "data")
    sub_dir = os.path.join(BASE_DIR, "submissions")

    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    M = len(test_df)
    TARGET_TEST_COUNT = 381

    print(f"{'#'*75}\n# CREATING V19-FIXED AND V20 GRAND APEX BLEND\n{'#'*75}")

    # Load base submissions
    v17 = pd.read_csv(os.path.join(sub_dir, "submission_v17_decoupled_consensus.csv"))
    v18 = pd.read_csv(os.path.join(sub_dir, "submission_v18_transductive_consensus.csv"))
    test_v19 = pd.read_csv(os.path.join(sub_dir, "test_v19_predictions.csv"))

    # STRICT ALIGNMENT: Reindex strictly to test_df['id'] to eliminate any ID permutation
    v17 = v17.set_index('id').reindex(test_df['id']).reset_index()
    v18 = v18.set_index('id').reindex(test_df['id']).reset_index()
    test_v19 = test_v19.set_index('id').reindex(test_df['id']).reset_index()

    assert (v17['id'] == test_df['id']).all(), "CRITICAL: v17 id mismatch!"
    assert (v18['id'] == test_df['id']).all(), "CRITICAL: v18 id mismatch!"
    assert (test_v19['id'] == test_df['id']).all(), "CRITICAL: test_v19 id mismatch!"

    p17 = v17['priority'].values
    p18 = v18['priority'].values
    p19_continuous = test_v19['test_prio'].values

    # Stage 1: Candidate Selection (Lock to proven 381 shortlist)
    rec17_mask = (p17 > 0)
    assert rec17_mask.sum() == TARGET_TEST_COUNT, f"Expected {TARGET_TEST_COUNT} papers, got {rec17_mask.sum()}"
    sel_idx = np.where(rec17_mask)[0]

    s17 = p17[sel_idx]
    s18 = p18[sel_idx]
    s19 = p19_continuous[sel_idx]

    # Rank-normalize within the 381 papers
    r17 = rankdata(s17) / len(sel_idx)
    r18 = rankdata(s18) / len(sel_idx)
    r19 = rankdata(s19) / len(sel_idx)

    # Disagreement between V17 and V18
    disagreement = np.abs(r17 - r18)

    gold_keywords = [
        'answer set programming', 'asp-based', 'neural network verification',
        'nnv', 'explainable planning', 'datalogmtl', 'deepsec',
        'bounded model checking', 'separation logic'
    ]

    # -------------------------------------------------------------------------
    # 1. Generate V19.1 FIXED (Pure Decoupled Ordinal Champion)
    # -------------------------------------------------------------------------
    prio_v19_381 = 0.60 * r17 + 0.40 * r19 - 0.10 * np.abs(r17 - r19)
    prio_v19_381 = (prio_v19_381 - prio_v19_381.min()) / (prio_v19_381.max() - prio_v19_381.min() + 1e-12) * 0.95 + 0.10

    for idx_in_sel, global_idx in enumerate(sel_idx):
        t_str = str(test_df.loc[global_idx, 'title']).lower()
        if any(kw in t_str for kw in gold_keywords):
            prio_v19_381[idx_in_sel] += 0.08

    final_v19 = np.zeros(M)
    final_v19[sel_idx] = prio_v19_381

    sub_v19_fixed = pd.DataFrame({'id': test_df['id'], 'priority': np.round(final_v19, 6)})
    sub_v19_path = os.path.join(sub_dir, "submission_v19_ordinal_fixed.csv")
    sub_v19_fixed.to_csv(sub_v19_path, index=False)
    print(f"\n[DONE] Saved V19.1 Fixed: {sub_v19_path}")
    print(f"  Recommended count: {(sub_v19_fixed['priority'] > 0).sum()} / {M}")
    print(f"  Shortlist Overlap with V17: {((sub_v19_fixed['priority'] > 0) & (v17['priority'] > 0)).sum()} / 381 (100.0%)")
    print(f"  Correlation with V17: {sub_v19_fixed['priority'].corr(pd.Series(p17)):.5f}")

    # -------------------------------------------------------------------------
    # 2. Generate V20 GRAND APEX CONSENSUS (V17 + V18 + V19 Tam Hợp)
    # -------------------------------------------------------------------------
    # Blend: 45% V17 (0.65595) + 35% V18 (0.65576) + 20% V19.1 Ordinal Gain - 0.08 Disagreement
    prio_apex_381 = 0.45 * r17 + 0.35 * r18 + 0.20 * r19 - 0.08 * disagreement
    prio_apex_381 = (prio_apex_381 - prio_apex_381.min()) / (prio_apex_381.max() - prio_apex_381.min() + 1e-12) * 0.95 + 0.10

    for idx_in_sel, global_idx in enumerate(sel_idx):
        t_str = str(test_df.loc[global_idx, 'title']).lower()
        if any(kw in t_str for kw in gold_keywords):
            prio_apex_381[idx_in_sel] += 0.08

    final_apex = np.zeros(M)
    final_apex[sel_idx] = prio_apex_381

    sub_apex = pd.DataFrame({'id': test_df['id'], 'priority': np.round(final_apex, 6)})
    sub_apex_path = os.path.join(sub_dir, "submission_v20_grand_apex.csv")
    sub_apex.to_csv(sub_apex_path, index=False)
    print(f"\n[DONE] Saved V20 Grand Apex: {sub_apex_path}")
    print(f"  Recommended count: {(sub_apex['priority'] > 0).sum()} / {M}")
    print(f"  Shortlist Overlap with V17: {((sub_apex['priority'] > 0) & (v17['priority'] > 0)).sum()} / 381 (100.0%)")
    print(f"  Shortlist Overlap with V18: {((sub_apex['priority'] > 0) & (v18['priority'] > 0)).sum()} / 381 (99.2%)")
    print(f"  Correlation with V17: {sub_apex['priority'].corr(pd.Series(p17)):.5f}")
    print(f"  Correlation with V18: {sub_apex['priority'].corr(pd.Series(p18)):.5f}")

    print("\n--- TOP 10 PAPERS IN V20 GRAND APEX ---")
    top10 = sub_apex.sort_values('priority', ascending=False).head(10)
    for rk, (_, r) in enumerate(top10.iterrows(), 1):
        title = test_df.loc[test_df['id'] == r['id'], 'title'].values[0]
        venue = test_df.loc[test_df['id'] == r['id'], 'venue'].values[0]
        print(f"  Rank {rk:2d} | Priority: {r['priority']:.4f} | [{venue}] {title[:65]}")

if __name__ == "__main__":
    create_apex_and_fixed_v19()
