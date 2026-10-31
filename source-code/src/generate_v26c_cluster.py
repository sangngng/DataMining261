import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SUB_DIR = Path("source-code/submissions")
DATA_DIR = Path("source-code/data")

def main():
    print("="*75)
    print("=== GENERATING V26C SURGICAL CLUSTER APEX (BUDGET 365) ===")
    print("="*75)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    v25b = pd.read_csv(SUB_DIR / "submission_v25b_transformer_hybrid_365.csv").set_index("id").reindex(test_df["id"]).reset_index()
    v20  = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test_df["id"]).reset_index()
    v26  = pd.read_csv(SUB_DIR / "test_v26_abstract_predictions.csv").set_index("id").reindex(test_df["id"]).reset_index()
    clusters = np.load(DATA_DIR / "clusters_k12.npz")
    te_c = clusters["te_clusters"]

    r_v25b = rankdata(v25b["priority"].values) / 596
    r_v20  = rankdata(v20["priority"].values) / 596
    r_v26_sel = rankdata(v26["v26_sel"].values) / 596
    r_v26_prio = rankdata(v26["v26_prio"].values) / 596

    delta = np.zeros(596, dtype=np.float64)
    delta[te_c == 8] += 0.08
    delta[te_c == 5] += 0.04
    delta[te_c == 9] -= 0.08

    final_sel = 0.55 * r_v25b + 0.30 * r_v26_sel + 0.15 * r_v20 + delta
    final_prio = 0.60 * r_v25b + 0.30 * r_v26_prio + 0.10 * r_v20 + delta

    budget = 365
    top_idx = np.argsort(final_sel)[::-1][:budget]

    out_p = np.zeros(len(test_df), dtype=np.float64)
    p_sel_top = final_prio[top_idx]
    out_p[top_idx] = p_sel_top - p_sel_top.min() + 0.05

    out_df = pd.DataFrame({
        "id": test_df["id"],
        "priority": np.round(out_p, 6)
    })

    out_name = "submission_v26c_cluster_boundary_365.csv"
    out_path = SUB_DIR / out_name
    out_df.to_csv(out_path, index=False)

    rec_count = (out_df["priority"] > 0).sum()
    overlap_v25b = ((out_df["priority"] > 0) & (v25b["priority"] > 0)).sum()
    corr_v25b = out_df["priority"].corr(v25b["priority"])

    print(f"\n[GENERATED] {out_name}")
    print(f"  Recommended papers: {rec_count}/596")
    print(f"  Shared with V25B (0.65947): {overlap_v25b}/{budget} ({overlap_v25b/budget*100:.2f}%)")
    print(f"  Priority Correlation with V25B: {corr_v25b:.5f}")
    print(f"  Papers modified/swapped: {budget - overlap_v25b}")

if __name__ == "__main__":
    main()
