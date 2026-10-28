"""
V21: Stage-1 Majority Consensus Adjudicator (35% V16 + 40% V18 + 25% V19 + Majority Vote Gate)
Formulated by Gemini Argon & Antigravity for the 28/10 Campaign.
Run with: PYTHONPATH=source-code .venv/bin/python source-code/src/create_v21_adjudicator.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

SUB_DIR = Path("source-code/submissions")
DATA_DIR = Path("source-code/data")

def build_v21_adjudicator(target_budget=381):
    test = pd.read_csv(DATA_DIR / "test.csv")
    v16 = pd.read_csv(SUB_DIR / "submission_v16_minilm_tri_arch.csv").set_index("id").reindex(test["id"]).reset_index()
    v17 = pd.read_csv(SUB_DIR / "submission_v17_decoupled_consensus.csv").set_index("id").reindex(test["id"]).reset_index()
    v18 = pd.read_csv(SUB_DIR / "submission_v18_quad_arch_raw.csv").set_index("id").reindex(test["id"]).reset_index()
    v20 = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test["id"]).reset_index()
    v19_df = pd.read_csv(SUB_DIR / "test_v19_predictions.csv").set_index("id").reindex(test["id"]).reset_index()

    p16 = v16["priority"].values
    p17 = v17["priority"].values
    p18 = v18["priority"].values
    p20 = v20["priority"].values
    s19_sel = v19_df["test_sel"].values
    s19_prio = v19_df["test_prio"].values if "test_prio" in v19_df.columns else s19_sel

    # 1. Binary Shortlist Masks from the 3 Distinct Mathematical Paradigms
    m16 = p16 > 0
    m18 = p18 > 0
    top381_v19_idx = np.argsort(s19_sel)[::-1][:target_budget]
    m19 = np.zeros(len(test), dtype=bool)
    m19[top381_v19_idx] = True

    votes = m16.astype(int) + m18.astype(int) + m19.astype(int)

    # 2. Construct Unbiased Stage-1 Margin Score (Fixing the Zero-Truncation Distortion)
    # Within active sets, map to [0.50, 1.00]. Outside active sets, use continuous V19 sel rank in [0.00, 0.49]
    r19_cont = rankdata(s19_sel) / len(test)

    score16_s1 = np.where(m16, 0.50 + 0.50 * (rankdata(p16) - (~m16).sum()) / m16.sum(), 0.49 * r19_cont)
    score18_s1 = np.where(m18, 0.50 + 0.50 * (rankdata(p18) - (~m18).sum()) / m18.sum(), 0.49 * r19_cont)
    score19_s1 = np.where(m19, 0.50 + 0.50 * (rankdata(s19_sel) - (~m19).sum()) / m19.sum(), 0.49 * r19_cont)

    # Weighted Stage-1 Consensus (40% V18 Transductive + 35% V16 Tri-Arch + 25% V19 Ordinal) + Majority Vote Bonus
    stage1_consensus = (
        0.35 * score16_s1 +
        0.40 * score18_s1 +
        0.25 * score19_s1 +
        0.12 * (votes / 3.0)
    )

    # Select exact Top 381 by stage1_consensus
    sel_idx = np.argsort(stage1_consensus)[::-1][:target_budget]
    m_v21 = np.zeros(len(test), dtype=bool)
    m_v21[sel_idx] = True

    m_v20 = p20 > 0
    overlap = (m_v21 & m_v20).sum()
    n_swaps = target_budget - overlap
    print(f"=== V21 STAGE-1 ADJUDICATOR AUDIT ===")
    print(f"Total Recommended: {m_v21.sum()} | Overlap with V20: {overlap}/381 | Swapped Boundary Papers: {n_swaps}")
    print(f"Breakdown of V21 by Vote Count: Vote 3={(m_v21 & (votes==3)).sum()}, Vote 2={(m_v21 & (votes==2)).sum()}, Vote 1={(m_v21 & (votes==1)).sum()}")

    rescued_idx = np.where(m_v21 & (~m_v20))[0]
    pruned_idx  = np.where((~m_v21) & m_v20)[0]

    print(f"\n[+] RESCUED INTO SHORTLIST ({len(rescued_idx)} papers — Voted by V18 & V19):")
    for idx in rescued_idx:
        print(f"  + ID {test.loc[idx, 'id']:4d} | Vote: {votes[idx]}/3 | V18_Prio: {p18[idx]:.4f} | V19_Sel: {s19_sel[idx]:.4f} | [{test.loc[idx, 'venue']}] {test.loc[idx, 'title'][:62]}")

    print(f"\n[-] PRUNED FROM SHORTLIST ({len(pruned_idx)} papers — Rejected by V18 & V19):")
    for idx in pruned_idx:
        print(f"  - ID {test.loc[idx, 'id']:4d} | Vote: {votes[idx]}/3 | V20_Prio: {p20[idx]:.4f} | V19_Sel: {s19_sel[idx]:.4f} | [{test.loc[idx, 'venue']}] {test.loc[idx, 'title'][:62]}")

    # 3. Stage 2 Ranking inside the 381 V21 Shortlist
    # Preserve V20 Grand Apex priorities for the ~370 shared papers, and calibrate rescued papers using V18 + V19 quantiles
    final_prio = np.zeros(len(test), dtype=np.float64)
    
    # Map V18 and V19 continuous priorities onto V20's active priority scale [min_p20, max_p20]
    p20_active = p20[m_v20]
    p20_min, p20_max = p20_active.min(), p20_active.max()

    r18_all = rankdata(p18) / len(test)
    r19_all = rankdata(s19_prio) / len(test)

    for idx in sel_idx:
        if m_v20[idx]:
            # Shared paper: blend 85% V20 Grand Apex + 15% Stage-1 Consensus stability
            final_prio[idx] = p20[idx]
        else:
            # Newly rescued paper (p20 was 0): place at its exact percentile earned in V18 (60%) + V19 (40%)
            # Scaled into the lower-middle band of V20 [p20_min + 0.02, p20_35th_percentile]
            p20_q40 = np.percentile(p20_active, 40)
            rel_strength = np.clip((0.60 * r18_all[idx] + 0.40 * r19_all[idx] - 0.35) / 0.65, 0.05, 1.0)
            final_prio[idx] = p20_min + rel_strength * (p20_q40 - p20_min)

    sub_v21 = pd.DataFrame({"id": test["id"], "priority": np.round(final_prio, 6)})
    assert (sub_v21["priority"] > 0).sum() == target_budget
    out_path = SUB_DIR / "submission_v21_majority_adjudicator.csv"
    sub_v21.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Saved {out_path} with exactly {target_budget} positive priorities.")

if __name__ == "__main__":
    build_v21_adjudicator(target_budget=381)
