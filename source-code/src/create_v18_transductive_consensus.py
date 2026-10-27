"""
V18 Transductive Consensus Blend:
Fuses V17 Decoupled Consensus (LB: 0.65595, current personal best) with
V18 Quad-Architecture Transductive Graph & Hard-Negative Boundary Model.
Strictly calibrated to the Golden Budget = 381 papers.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SUB_DIR = BASE_DIR / "submissions"
DATA_DIR = BASE_DIR / "data"

def create_v18_consensus_submission():
    v17_path = SUB_DIR / "submission_v17_decoupled_consensus.csv"
    v18_path = SUB_DIR / "submission_v18_decoupled_consensus.csv"
    test_path = DATA_DIR / "test.csv"

    v17 = pd.read_csv(v17_path).sort_values("id").reset_index(drop=True)
    v18 = pd.read_csv(v18_path).sort_values("id").reset_index(drop=True)
    test = pd.read_csv(test_path).sort_values("id").reset_index(drop=True)

    print(f"Loaded V17 (LB: 0.65595, non-zero: {(v17['priority']>0).sum()})")
    print(f"Loaded V18 (Quad-Arch Transductive, non-zero: {(v18['priority']>0).sum()})")

    p17 = v17["priority"].values.copy()
    p18 = v18["priority"].values.copy()

    # 50/50 Balanced Consensus Blend of normalized priorities
    w18 = 0.50
    blend_raw = (1.0 - w18) * p17 + w18 * p18

    # Strictly calibrate to target budget = 381 papers (63.93%)
    TARGET_BUDGET = 381
    top381_idx = np.argsort(blend_raw)[::-1][:TARGET_BUDGET]

    final_priorities = np.zeros(len(test), dtype=np.float64)
    final_priorities[top381_idx] = blend_raw[top381_idx]

    sub_v18_consensus = pd.DataFrame({
        "id": test["id"],
        "priority": np.round(final_priorities, 6)
    })

    out_path = SUB_DIR / "submission_v18_transductive_consensus.csv"
    sub_v18_consensus.to_csv(out_path, index=False)

    print(f"\n[SUCCESS] Generated: {out_path}")
    print(f"Total rows: {len(sub_v18_consensus)}")
    print(f"Non-zero count: {(sub_v18_consensus['priority'] > 0).sum()} / {len(sub_v18_consensus)} ({(sub_v18_consensus['priority'] > 0).mean()*100:.2f}%)")
    print(f"Min positive priority: {sub_v18_consensus[sub_v18_consensus['priority'] > 0]['priority'].min():.4f}")
    print(f"Max priority: {sub_v18_consensus['priority'].max():.4f}")
    print(f"Correlation with V17 (0.65595): {sub_v18_consensus['priority'].corr(v17['priority']):.5f}")
    print(f"Correlation with V18 Raw: {sub_v18_consensus['priority'].corr(v18['priority']):.5f}")

    # Check key domain papers
    for title_kw in ["DatalogMTL", "DEEPSEC", "Explainable Planning Using Answer Set"]:
        matches = test[test["title"].str.contains(title_kw, case=False)]
        for _, r in matches.iterrows():
            p_val = sub_v18_consensus.loc[sub_v18_consensus["id"] == r["id"], "priority"].values[0]
            print(f"  * [{r['venue']}] {r['title'][:60]} -> Priority: {p_val:.4f} (Recommended: {p_val > 0})")

    return out_path

if __name__ == "__main__":
    create_v18_consensus_submission()
