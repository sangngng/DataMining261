"""
V25A: Immediate Part-2 Distribution & Budget Calibration (342 Recs) + DOI Publisher Signal
Run with: /Users/sanggnguyenn/Documents/HCMUT/DataMining/.venv/bin/python source-code/src/create_v25a_part2_calibrated.py
"""
import sys
sys.path.append("source-code")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path
from src.metric import compute_competition_metric

DATA_DIR = Path("source-code/data")
SUB_DIR = Path("source-code/submissions")

def extract_doi_and_venue_part2_bonus(df):
    """
    Applies verified Part-2 (id >= 713) empirical likelihoods from DOI prefix and Venue.
    """
    dois = df["doi"].fillna("").astype(str).str.lower()
    venues = df["venue"].fillna("").astype(str).str.lower()

    bonus = np.zeros(len(df), dtype=np.float64)
    # Verified DOI source signals from Root Cause Audit
    bonus += np.where(dois.str.contains("10.24963"), 0.08, 0.0)       # 87.9% positive rate
    bonus += np.where(dois.str.contains("ceur"), 0.07, 0.0)           # 80.8% positive rate
    bonus += np.where(dois.str.contains("10.1007"), 0.03, 0.0)        # 69.2% positive rate
    bonus -= np.where(dois.str.contains("semanticscholar"), 0.05, 0.0)  # 49.8% positive rate
    bonus -= np.where(dois.str.contains("10.1109"), 0.03, 0.0)        # 53.1% positive rate

    # Verified Part-2 Venue prior adjustments (relative to Part-1 skewed training)
    # In Part 2: lpnmr (76.7%), kr (74.8%) vs lics (53.3%), cav (52.5%)
    bonus += np.where(venues == "lpnmr", 0.05, 0.0)
    bonus += np.where(venues == "kr", 0.04, 0.0)
    bonus -= np.where(venues == "cav", 0.03, 0.0)
    bonus -= np.where(venues == "lics", 0.02, 0.0)
    return bonus

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    # 1. Verify OOF gain on Part 2 (id >= 713) when calibrating budget & DOI/Venue prior
    oof_v16 = pd.read_csv(SUB_DIR / "oof_v16_minilm_tri_arch.csv")
    oof_v18 = pd.read_csv(SUB_DIR / "oof_v18_quad_arch.csv")
    oof_v24 = pd.read_csv(SUB_DIR / "oof_v24_harmonic_community.csv")

    part2_mask = (train["id"] >= 713).values
    p2_df = train.loc[part2_mask].reset_index(drop=True)

    c16 = "oof_prob_ens" if "oof_prob_ens" in oof_v16.columns else [c for c in oof_v16.columns if c not in ["id", "Label"]][0]
    c18 = "oof_prob_ens" if "oof_prob_ens" in oof_v18.columns else [c for c in oof_v18.columns if c not in ["id", "Label"]][0]

    r16_oof = rankdata(oof_v16[c16].values[part2_mask]) / len(p2_df)
    r18_oof = rankdata(oof_v18[c18].values[part2_mask]) / len(p2_df)
    r24_oof = rankdata(oof_v24["s_harmonic"].values[part2_mask]) / len(p2_df)
    doi_bonus_oof = extract_doi_and_venue_part2_bonus(p2_df)

    fused_oof_sel = 0.40 * r16_oof + 0.35 * r18_oof + 0.25 * r24_oof + doi_bonus_oof
    fused_oof_prio = 0.65 * rankdata(oof_v16[c16].values[part2_mask]) / len(p2_df) + 0.35 * rankdata(oof_v24["prio_v24"].values[part2_mask]) / len(p2_df) + 0.5 * doi_bonus_oof

    print("="*75)
    print("=== PART-2 (TEST-MATCHED) OOF AUDIT WITH DOI/VENUE CALIBRATION ===")
    print("="*75)
    for test_budget in [381, 360, 350, 344, 342, 340, 335, 330]:
        q_pct = test_budget / 596.0
        th = np.quantile(fused_oof_sel, 1.0 - q_pct)
        sub_p = np.where(fused_oof_sel >= th, fused_oof_prio + 0.10, 0.0)
        sc, Q, R = compute_competition_metric(p2_df["id"].values, p2_df["Label"].values, sub_p)
        print(f"  Target Test Budget = {test_budget:3d}/596 ({q_pct*100:5.2f}%) -> Part-2 CV Score: {sc:.5f} | Q: {Q:.5f} | R: {R:.5f}")

    # 2. Build Test Submission V25A (Calibrated Budget = 342 and 352 papers)
    v20 = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test["id"]).reset_index()
    v18 = pd.read_csv(SUB_DIR / "submission_v18_quad_arch_raw.csv").set_index("id").reindex(test["id"]).reset_index()
    v19 = pd.read_csv(SUB_DIR / "test_v19_predictions.csv").set_index("id").reindex(test["id"]).reset_index()
    v24 = pd.read_csv(SUB_DIR / "test_v24_predictions.csv").set_index("id").reindex(test["id"]).reset_index()

    r20 = rankdata(v20["priority"].values) / len(test)
    r18 = rankdata(v18["priority"].values) / len(test)
    r19 = rankdata(v19["test_sel"].values) / len(test)
    r24 = rankdata(v24["s_harmonic"].values) / len(test)
    doi_bonus_test = extract_doi_and_venue_part2_bonus(test)

    # Stage 1 Selection Score anchored on V20 + V18 + V19 + V24 + Part-2 DOI/Venue Prior
    sel_score_test = 0.35 * r20 + 0.25 * r18 + 0.15 * r19 + 0.25 * r24 + doi_bonus_test

    # Stage 2 Priority Score inside selected shortlist: preserve V20 Grand Apex (80%) + V24 (20%)
    r24_prio = rankdata(v24["prio_v24"].values) / len(test)
    p20_vals = v20["priority"].values
    stage2_prio = np.where(p20_vals > 0, p20_vals + 0.12 * r24_prio + 0.25 * doi_bonus_test, 0.50 + 0.40 * r24_prio + 0.25 * doi_bonus_test)

    for target_budget in [342, 352]:
        top_idx = np.argsort(sel_score_test)[::-1][:target_budget]
        out_prio = np.zeros(len(test), dtype=np.float64)
        out_prio[top_idx] = stage2_prio[top_idx] - stage2_prio[top_idx].min() + 0.05

        sub = pd.DataFrame({"id": test["id"], "priority": np.round(out_prio, 6)})
        fname = f"submission_v25a_part2_budget_{target_budget}.csv"
        sub.to_csv(SUB_DIR / fname, index=False)
        overlap_v20 = ((sub["priority"] > 0) & (v20["priority"] > 0)).sum()
        corr_v20 = sub["priority"].corr(v20["priority"])
        print(f"\n[SAVED] {fname}")
        print(f"  Recs: {target_budget}/596 | Shared with V20 (381): {overlap_v20}/{target_budget} ({overlap_v20/target_budget*100:.1f}%)")
        print(f"  Correlation with V20: {corr_v20:.5f}")

if __name__ == "__main__":
    main()
