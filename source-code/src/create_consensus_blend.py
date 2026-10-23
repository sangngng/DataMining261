import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from src.utils import save_submission

def create_consensus_blend():
    print("--- Creating Consensus Gated Master Blend (V2 + V4 + V5) ---")
    sub_dir = os.path.join(BASE_DIR, "submissions")
    
    v2_path = os.path.join(sub_dir, "submission_v2_sbert_ensemble.csv")
    v4_path = os.path.join(sub_dir, "submission_v4_clean_ensemble.csv")
    v5_path = os.path.join(sub_dir, "submission_v5_dual_embedding.csv")
    
    v2 = pd.read_csv(v2_path)
    v4 = pd.read_csv(v4_path)
    v5 = pd.read_csv(v5_path)
    
    # 1. Soft Priority Blending: 35% V2 + 35% V4 + 30% V5
    weighted_p = 0.35 * v2['priority'] + 0.35 * v4['priority'] + 0.30 * v5['priority']
    
    # 2. Hard Majority Consensus Gate:
    # A paper is recommended if and only if AT LEAST 2 out of the 3 top models recommend it (priority > 0).
    votes = (v2['priority'] > 0).astype(int) + (v4['priority'] > 0).astype(int) + (v5['priority'] > 0).astype(int)
    
    # Apply consensus filter
    final_priorities = np.where(votes >= 2, weighted_p, 0.0)
    final_priorities = np.clip(final_priorities, 0.0, None)
    
    recom_count = int(np.sum(final_priorities > 0))
    pct = 100.0 * recom_count / len(final_priorities)
    print(f"Total Test Papers: {len(final_priorities)}")
    print(f"Recommended Count (Votes >= 2): {recom_count} ({pct:.1f}%)")
    print(f"Filtered Out (Votes <= 1): {len(final_priorities) - recom_count} ({100.0 - pct:.1f}%)")
    
    out_file = save_submission(
        ids=v2['id'].tolist(),
        priorities=final_priorities.tolist(),
        filename="submission_v6_consensus_blend.csv",
        output_dir=sub_dir
    )
    
    print(f"[SUCCESS] Saved Consensus Master Blend to: {out_file}")
    return out_file

if __name__ == "__main__":
    create_consensus_blend()
