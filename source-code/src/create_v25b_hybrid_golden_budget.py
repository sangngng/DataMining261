"""
Create Golden Budget Submissions for V25B Hybrid:
Budgets: 360, 365, 370
Blends: Transformer V25B (45%) + V25A (35%) + V20 Grand Apex (20%)
"""
import sys
sys.path.append("source-code")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

DATA_DIR = Path("source-code/data")
SUB_DIR = Path("source-code/submissions")

def generate_golden_budget_submissions():
    test = pd.read_csv(DATA_DIR / "test.csv")
    v25b = pd.read_csv(SUB_DIR / "test_v25b_transformer_predictions.csv").set_index("id").reindex(test["id"]).reset_index()
    v25a = pd.read_csv(SUB_DIR / "submission_v25a_part2_budget_342.csv").set_index("id").reindex(test["id"]).reset_index()
    v20  = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test["id"]).reset_index()

    r_trans_sel = rankdata(v25b["v25b_sel"].values) / len(test)
    r_v20_sel   = rankdata(v20["priority"].values) / len(test)
    r_v25a_sel  = rankdata(v25a["priority"].values) / len(test)

    # Stage 1 Selection Score
    final_sel = 0.45 * r_trans_sel + 0.35 * r_v25a_sel + 0.20 * r_v20_sel
    
    # Stage 2 Priority Score
    final_prio = 0.45 * (rankdata(v25b["v25b_prio"].values) / len(test)) + 0.55 * (rankdata(v20["priority"].values) / len(test))

    for target_budget in [360, 365, 370]:
        top_idx = np.argsort(final_sel)[::-1][:target_budget]
        out_p = np.zeros(len(test), dtype=np.float64)
        out_p[top_idx] = final_prio[top_idx] - final_prio[top_idx].min() + 0.05
        sub = pd.DataFrame({"id": test["id"], "priority": np.round(out_p, 6)})
        out_name = f"submission_v25b_transformer_hybrid_{target_budget}.csv"
        sub.to_csv(SUB_DIR / out_name, index=False)
        overlap_v20 = ((sub["priority"] > 0) & (v20["priority"] > 0)).sum()
        corr_v20 = sub["priority"].corr(v20["priority"])
        print(f"[SAVED] {out_name} | Recommended: {target_budget}/596 | Shared with V20: {overlap_v20}/{target_budget} ({overlap_v20/target_budget*100:.1f}%) | Corr with V20: {corr_v20:.5f}")

if __name__ == "__main__":
    generate_golden_budget_submissions()
