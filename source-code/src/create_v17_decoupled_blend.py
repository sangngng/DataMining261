"""
V17: Decoupled Stage-1 Selection (V16) + Stage-2 Consensus Ranking (V16 + V13 + V8)
Run with: PYTHONPATH=source-code .venv/bin/python source-code/src/create_v17_decoupled_blend.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SUB_DIR = BASE_DIR / "submissions"
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(BASE_DIR))
from src.metric import compute_competition_metric

def load_and_verify():
    v16 = pd.read_csv(SUB_DIR / "submission_v16_minilm_tri_arch.csv").sort_values("id").reset_index(drop=True)
    v13 = pd.read_csv(SUB_DIR / "submission_v13_deep_domain.csv").sort_values("id").reset_index(drop=True)
    v8  = pd.read_csv(SUB_DIR / "submission_v8_pure_semantic_blend.csv").sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(DATA_DIR / "test.csv").sort_values("id").reset_index(drop=True)

    assert len(v16) == 596 and len(v13) == 596, "Row count mismatch!"
    
    m16 = v16["priority"] > 0
    m13 = v13["priority"] > 0
    m8  = v8["priority"] > 0

    shared = (m16 & m13).sum()
    v16_only = (m16 & ~m13).sum()
    v13_only = (~m16 & m13).sum()
    print(f"[Audit] V16 recs: {m16.sum()} | V13 recs: {m13.sum()} | Shared: {shared} | V16-only: {v16_only} | V13-only: {v13_only}")
    return v16, v13, v8, test_df

def build_v17_decoupled_blend(v16, v13, v8, test_df, target_budget=381, enable_swaps=False):
    """
    Theorem:
    1. Stage 1 Selection: Anchor on V16's 381 shortlist (0.85795 Coverage), 
       which brought us to Rank 4 (0.65474).
       If enable_swaps=True, allows up to 2 high-confidence consensus rescues.
       If enable_swaps=False, locks 100% of V16's 381 shortlist and only re-orders for Q.
    2. Stage 2 Ranking: Inside the 381 shortlist, combine V16's 10-fold Tri-Arch rank
       with V13's high-Q domain-calibrated priority so 5-star papers lock into Top 50 DCG.
    """
    p16 = v16["priority"].values.copy()
    p13 = v13["priority"].values.copy()
    p8  = v8["priority"].values.copy()

    # --- STEP 1: SHORTLIST SELECTION (381 PAPERS) ---
    selected_mask = (p16 > 0).copy()
    
    if enable_swaps:
        # Find V16's weakest selected papers (ranks 377..381)
        v16_pos_indices = np.where(p16 > 0)[0]
        v16_pos_sorted = v16_pos_indices[np.argsort(p16[v16_pos_indices])] # ascending (weakest first)
        
        # Find V13's strongest excluded papers (where p16 == 0 but p13 > 0 and p8 > 0)
        v13_only_indices = np.where((p16 == 0) & (p13 > 0) & (p8 > 0))[0]
        if len(v13_only_indices) > 0:
            v13_active_vals = p13[p13 > 0]
            v13_q50 = np.percentile(v13_active_vals, 50) # Must be top 50% of V13 active
            strong_v13_rescues = [idx for idx in v13_only_indices if p13[idx] >= v13_q50]
            strong_v13_rescues = sorted(strong_v13_rescues, key=lambda idx: p13[idx], reverse=True)[:2]
            
            v16_only_weak = [idx for idx in v16_pos_sorted if p13[idx] == 0][:len(strong_v13_rescues)]
            
            print(f"\n[Boundary Audit] Found {len(strong_v13_rescues)} high-confidence V13+V8 papers missed by V16:")
            for r_idx, w_idx in zip(strong_v13_rescues, v16_only_weak):
                print(f"  + Rescue ID={test_df.loc[r_idx, 'id']} ({test_df.loc[r_idx, 'venue']}) | V13_prio={p13[r_idx]:.4f} | Title: {test_df.loc[r_idx, 'title'][:65]}...")
                print(f"  - Drop   ID={test_df.loc[w_idx, 'id']} ({test_df.loc[w_idx, 'venue']}) | V16_prio={p16[w_idx]:.4f} | Title: {test_df.loc[w_idx, 'title'][:65]}...")
                selected_mask[r_idx] = True
                selected_mask[w_idx] = False

    assert selected_mask.sum() == target_budget, f"Expected {target_budget}, got {selected_mask.sum()}"

    # --- STEP 2: INTRA-SHORTLIST DCG RE-RANKING (MAXIMIZING Q) ---
    sel_idx = np.where(selected_mask)[0]
    
    # Normalize V16 priorities among selected papers
    s16_raw = p16[sel_idx]
    min_active_16 = np.percentile(s16_raw[s16_raw > 0], 15) if (s16_raw > 0).any() else 0.5
    s16_raw = np.where(s16_raw > 0, s16_raw, min_active_16)
    s16_minmax = (s16_raw - s16_raw.min()) / (s16_raw.max() - s16_raw.min() + 1e-12)
    s16_rank   = rankdata(s16_raw) / len(sel_idx)
    score_16   = 0.60 * s16_minmax + 0.40 * s16_rank

    # Normalize V13 priorities among selected papers
    s13_raw = p13[sel_idx]
    s13_pos_mask = s13_raw > 0
    s13_norm = np.zeros_like(s13_raw)
    if s13_pos_mask.sum() > 0:
        pos_vals = s13_raw[s13_pos_mask]
        pos_minmax = (pos_vals - pos_vals.min()) / (pos_vals.max() - pos_vals.min() + 1e-12)
        pos_rank   = rankdata(pos_vals) / len(pos_vals)
        s13_norm[s13_pos_mask] = 0.65 * pos_minmax + 0.35 * pos_rank
    # Impute V16-only papers at 0.85 * their V16 score
    s13_norm[~s13_pos_mask] = 0.85 * score_16[~s13_pos_mask]

    # Consensus Disagreement Penalty
    disagreement = np.abs(score_16 - s13_norm)

    # Blended Priority inside 381 shortlist:
    # 65% V16 Tri-Arch + 35% V13 Deep-Domain - 0.12 Disagreement Penalty
    blended_sel_prio = 0.65 * score_16 + 0.35 * s13_norm - 0.12 * disagreement
    blended_sel_prio = blended_sel_prio - blended_sel_prio.min() + 0.10

    final_prio = np.zeros(len(v16), dtype=np.float64)
    final_prio[sel_idx] = blended_sel_prio

    variant_name = "submission_v17_decoupled_consensus.csv" if not enable_swaps else "submission_v17_swapped_consensus.csv"
    sub_v17 = pd.DataFrame({"id": v16["id"], "priority": np.round(final_prio, 6)})
    out_path = SUB_DIR / variant_name
    sub_v17.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Generated {out_path}")
    print(f"Non-zero count: {(sub_v17['priority'] > 0).sum()} / {len(sub_v17)} ({(sub_v17['priority'] > 0).mean()*100:.2f}%)")
    print(f"Correlation with V16: {sub_v17['priority'].corr(v16['priority']):.5f}")
    print(f"Correlation with V13: {sub_v17['priority'].corr(v13['priority']):.5f}")
    return sub_v17

if __name__ == "__main__":
    v16, v13, v8, test_df = load_and_verify()
    print("\n--- Generating V17 Option 1 (Pure V16 Shortlist 100% Preserved + V13 DCG Re-Rank) ---")
    sub17_pure = build_v17_decoupled_blend(v16, v13, v8, test_df, target_budget=381, enable_swaps=False)
    print("\n--- Generating V17 Option 2 (Boundary Rescue Audit) ---")
    sub17_swap = build_v17_decoupled_blend(v16, v13, v8, test_df, target_budget=381, enable_swaps=True)
