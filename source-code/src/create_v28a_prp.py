import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

DATA_DIR = Path("source-code/data")
SUB_DIR  = Path("source-code/submissions")

def generate_v28a():
    print("="*75)
    print("=== GENERATING V28A PURE DECOUPLED PRP SUBMISSION ===")
    print("="*75)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_abs = pd.read_csv(DATA_DIR / "test_with_abstracts.csv").set_index("id").reindex(test_df["id"]).reset_index()
    clusters = np.load(DATA_DIR / "clusters_k12.npz")
    te_c = clusters["te_clusters"]

    v26c = pd.read_csv(SUB_DIR / "submission_v26c_cluster_boundary_365.csv").set_index("id").reindex(test_df["id"]).reset_index()
    p19  = pd.read_csv(SUB_DIR / "test_v19_predictions.csv").set_index("id").reindex(test_df["id"]).reset_index()
    p26  = pd.read_csv(SUB_DIR / "test_v26_abstract_predictions.csv").set_index("id").reindex(test_df["id"]).reset_index()
    p27  = pd.read_csv(SUB_DIR / "test_v27_abstract_predictions.csv").set_index("id").reindex(test_df["id"]).reset_index()

    # 1. Identify active 365 papers from V26C (100% locked boundary)
    active_mask = v26c["priority"].values > 0
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)
    assert n_active == 365, f"Expected 365 active papers, got {n_active}"

    # V26C overall rank (1 = highest priority, 596 = lowest)
    v26c_rank = 597 - rankdata(v26c["priority"].values)

    # 2. Compute E[g] across active 365 papers
    # Blend V26 priority, V27 priority, and V19 Frank-Hall convex ordinal gain
    r_p26 = rankdata(p26.loc[active_mask, "v26_prio"].values) / n_active
    r_p27 = rankdata(p27.loc[active_mask, "v27_prio"].values) / n_active
    r_p19 = rankdata(p19.loc[active_mask, "test_prio"].values) / n_active
    e_g = 0.50 * r_p26 + 0.30 * r_p27 + 0.20 * r_p19
    r_eg = rankdata(e_g) / n_active

    # NormRank of V26C priority within active set
    r_v26c_active = rankdata(v26c.loc[active_mask, "priority"].values) / n_active

    # 3. Tiered Domain & Author Boost
    tier1_authors = ["tran cao son", "torsten schaub", "marcello balduccini", "pedro cabalar"]
    tier2_authors = ["francesco ricca", "fabrizio riguzzi", "thomas eiter", "michael gelfond", 
                     "vladimir lifschitz", "wolfgang faber", "taylor t. johnson", "hoang-dung tran"]

    asp_keywords = ["answer set programming", "clingo", "dlv", "equilibrium logic", "epistemic specifications", "asp-based", "xasp", "alaspo"]
    nnv_keywords = ["nnv", "marabou", "star sets", "zonotope", "neural network verification", "neural networks verification"]

    beta_auth = np.zeros(n_active, dtype=np.float64)
    beta_topic = np.zeros(n_active, dtype=np.float64)

    for i, g_idx in enumerate(active_idx):
        row = test_df.iloc[g_idx]
        a_str = str(row["authors"]).lower()
        t_str = str(row["title"]).lower()
        v_str = str(row["venue"]).lower()
        abs_str = str(test_abs.loc[g_idx, "abstract"]).lower() if "abstract" in test_abs.columns else ""
        c_label = te_c[g_idx]

        # Author Boost
        if any(ta in a_str for ta in tier1_authors):
            beta_auth[i] = 0.15
        elif any(ta in a_str for ta in tier2_authors):
            beta_auth[i] = 0.10

        # Topic Boost
        is_asp = (c_label == 8) or any(kw in t_str or kw in abs_str for kw in asp_keywords)
        is_nnv = (c_label == 5) or any(kw in t_str or kw in abs_str for kw in nnv_keywords)

        if is_asp and (v_str in ["lpnmr", "kr", "iclp"]):
            beta_topic[i] = 0.10
        elif is_nnv and (v_str in ["cav", "lics"]):
            beta_topic[i] = 0.08

    # 4. Gated Ranking Score: only papers with V26C rank <= 120 receive the boost
    gated_condition = (v26c_rank[active_idx] <= 120).astype(np.float64)
    score_stage2 = 0.55 * r_eg + 0.45 * r_v26c_active + gated_condition * (beta_auth + beta_topic)

    # 5. Build final priority array
    final_prio = np.zeros(len(test_df), dtype=np.float64)
    # Calibrate scale: min active priority ~ 0.05, max ~ 1.25
    scaled_prio = (score_stage2 - score_stage2.min()) / (score_stage2.max() - score_stage2.min() + 1e-12) * 0.95 + 0.05
    final_prio[active_idx] = np.round(scaled_prio, 6)

    out_df = pd.DataFrame({
        "id": test_df["id"],
        "priority": final_prio
    })

    out_name = "submission_v28a_pure_decoupled_prp_365.csv"
    out_path = SUB_DIR / out_name
    out_df.to_csv(out_path, index=False)

    print(f"\n[GENERATED] {out_name}")
    print(f"  Recommended count: {(out_df['priority'] > 0).sum()}/596")
    print(f"  Exact overlap with V26C (0.66297): {((out_df['priority'] > 0) & (v26c['priority'] > 0)).sum()}/365 (100.0%)")
    print(f"  Swapped boundary papers: 0")
    print(f"  Priority correlation with V26C: {out_df['priority'].corr(v26c['priority']):.5f}")

    # Inspect Top 25
    m = out_df.merge(test_df, on="id").sort_values("priority", ascending=False).reset_index(drop=True)
    print("\n" + "="*75)
    print("=== TOP 25 PAPERS IN V28A ===")
    print("="*75)
    for i in range(25):
        r = m.iloc[i]
        old_r = int(v26c_rank[test_df["id"] == r["id"]][0])
        print(f"{i+1:2d}. [Old V26C Rank: {old_r:3d}] ID {r['id']:4d} [{r['venue']:5s}] {r['title'][:60]} (p={r['priority']:.4f})")

    # Specifically check Star Authors
    print("\n" + "="*75)
    print("=== STAR AUTHOR PAPERS POSITION IN V28A VS V26C ===")
    print("="*75)
    star_check_ids = [979, 1296, 1343, 1395, 1330, 1313, 1337, 1317, 1305, 1378]
    m_full = out_df.merge(test_df, on="id")
    m_full["v28a_rank"] = 597 - rankdata(m_full["priority"])
    m_full["v26c_rank"] = v26c_rank
    for pid in star_check_ids:
        r = m_full[m_full["id"] == pid].iloc[0]
        print(f"ID {pid:4d} [{r['venue']:5s}]: V28A Rank={int(r['v28a_rank']):2d} (was V26C Rank={int(r['v26c_rank']):2d}) | {r['title'][:60]}")

if __name__ == "__main__":
    generate_v28a()
