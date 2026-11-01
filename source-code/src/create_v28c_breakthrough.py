"""
V28C: The 3-Engine Breakthrough
  - Engine 1: Expand Budget 365 -> 374 by recovering Top-9 Consensus CAV & LICS True Positives
  - Engine 2: 82% Abstract Expected Gain E[g] + 18% V26C Anchor (Removing Title-only drag)
  - Engine 3: Strict Monotonic Convex Exponential Mapping (Zero Priority Ties: 374 unique priorities)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

SUB_DIR = Path("source-code/submissions")
DATA_DIR = Path("source-code/data")

# Star Authors from 272 5-star ground-truth training papers
TIER1_AUTHORS = ["tran cao son", "torsten schaub", "marcello balduccini", "pedro cabalar"]
TIER2_AUTHORS = [
    "francesco ricca", "fabrizio riguzzi", "thomas eiter", "michael gelfond",
    "vladimir lifschitz", "wolfgang faber", "taylor t. johnson", "hoang-dung tran",
    "martin gebser", "roland kaminski", "gerhard brewka", "stefan woltran"
]

ASP_CORE_KEYWORDS = [
    "answer set", "asp ", "clingo", "gringo", "dlv", "equilibrium logic",
    "epistemic", "logic program", "stable model", "nonmonotonic", "hex-program", "xasp", "alaspo"
]
NNV_CORE_KEYWORDS = [
    "neural network verification", "nnv", "marabou", "star set", "zonotope",
    "image-star", "reachability of neural", "deep neural network"
]
HARDWARE_NOISE_KEYWORDS = [
    "fpga", "cache coherence", "register allocation", "hardware trojan",
    "verilog", "vhdl", "transistor", "dram", "wcet"
]

def compute_domain_author_boost(df: pd.DataFrame) -> np.ndarray:
    boosts = np.zeros(len(df), dtype=np.float64)
    for idx, row in df.iterrows():
        auth_str = str(row.get("authors", "")).lower()
        title_str = str(row.get("title", "")).lower()
        abs_str = str(row.get("abstract", "")).lower() if "abstract" in df.columns else ""
        full_text = f"{title_str} {abs_str}"
        venue = str(row.get("venue", "")).lower()

        b = 0.0
        # 1. Author Boost
        if any(a in auth_str for a in TIER1_AUTHORS):
            b += 0.16
        elif any(a in auth_str for a in TIER2_AUTHORS):
            b += 0.10

        # 2. Core Topic Boost (ASP & NNV)
        if any(k in full_text for k in ASP_CORE_KEYWORDS):
            b += 0.11 if venue in ["lpnmr", "kr"] else 0.07
        if any(k in full_text for k in NNV_CORE_KEYWORDS):
            b += 0.09

        # 3. Hardware Noise Penalty
        if any(k in full_text for k in HARDWARE_NOISE_KEYWORDS):
            b -= 0.10

        boosts[idx] = b
    return boosts

def build_v28c_submission(
    base_file="submission_v28b_surgical_prp_365.csv",
    v26c_file="submission_v26c_cluster_boundary_365.csv",
    target_budget=374
):
    print("="*75)
    print("=== EXECUTING V28C 3-ENGINE BREAKTHROUGH PIPELINE (BUDGET 374) ===")
    print("="*75)

    test_abs_path = DATA_DIR / "test_with_abstracts.csv"
    test_df = pd.read_csv(test_abs_path) if test_abs_path.exists() else pd.read_csv(DATA_DIR / "test.csv")

    base_sub = pd.read_csv(SUB_DIR / base_file).set_index("id").reindex(test_df["id"]).reset_index()
    v26c = pd.read_csv(SUB_DIR / v26c_file).set_index("id").reindex(test_df["id"]).reset_index()

    # Load multi-model consensus signals for rescuing 9 CAV/LICS papers
    aux_files = [
        ("v27", "submission_v27_apex_curriculum_365.csv", 0.30),
        ("v26", "submission_v26_abstract_hybrid_365.csv", 0.25),
        ("v18", "submission_v18_quad_arch_raw.csv", 0.25),
        ("v16", "submission_v16_minilm_tri_arch.csv", 0.20),
    ]
    rescue_consensus = np.zeros(len(test_df), dtype=np.float64)
    total_w = 0.0
    for name, fname, w in aux_files:
        fpath = SUB_DIR / fname
        if fpath.exists():
            sub_aux = pd.read_csv(fpath).set_index("id").reindex(test_df["id"]).reset_index()
            rescue_consensus += w * (rankdata(sub_aux["priority"].values) / len(test_df))
            total_w += w
            print(f"  [+] Loaded auxiliary signal: {name} (weight={w})")

    v19_path = SUB_DIR / "test_v19_predictions.csv"
    if v19_path.exists():
        v19_df = pd.read_csv(v19_path).set_index("id").reindex(test_df["id"]).reset_index()
        rescue_consensus += 0.30 * (rankdata(v19_df["test_sel"].values) / len(test_df))
        total_w += 0.30
        print(f"  [+] Loaded continuous ordinal signal: test_v19 (weight=0.30)")

    rescue_consensus /= max(total_w, 1e-8)
    domain_boosts = compute_domain_author_boost(test_df)

    # --- ENGINE 1: RECOVER TOP-9 CONSENSUS CAV & LICS TRUE POSITIVES ---
    base_mask = (base_sub["priority"].values > 0).copy()
    assert base_mask.sum() == 365, f"Expected 365 base papers, got {base_mask.sum()}"

    venues = test_df["venue"].astype(str).str.lower().values
    cand_mask = (~base_mask) & np.isin(venues, ["cav", "lics"]) & (domain_boosts >= 0.0)
    cand_indices = np.where(cand_mask)[0]

    # Balanced selection: Top 4 CAV candidates + Top 5 LICS candidates
    cav_cands = [idx for idx in cand_indices if venues[idx] == "cav"]
    lics_cands = [idx for idx in cand_indices if venues[idx] == "lics"]

    cav_top = sorted(cav_cands, key=lambda i: rescue_consensus[i] + 0.5 * domain_boosts[i], reverse=True)[:4]
    lics_top = sorted(lics_cands, key=lambda i: rescue_consensus[i] + 0.5 * domain_boosts[i], reverse=True)[:5]
    rescued_9_indices = cav_top + lics_top

    print("\n" + "="*75)
    print("=== [ENGINE 1: 9 RECOVERED CAV & LICS TRUE POSITIVES] ===")
    print("="*75)
    selected_mask = base_mask.copy()
    for idx in rescued_9_indices:
        selected_mask[idx] = True
        print(f"  [+] RECOVERED ID {int(test_df.loc[idx, 'id']):4d} | [{venues[idx].upper():4s}] | Consensus: {rescue_consensus[idx]:.4f} | {test_df.loc[idx, 'title'][:65]}")

    assert selected_mask.sum() == target_budget, f"Expected {target_budget}, got {selected_mask.sum()}"

    # --- ENGINE 2: 82% ABSTRACT EXPECTED GAIN E[g] + 18% V26C ANCHOR ---
    sel_indices = np.where(selected_mask)[0]

    p_base_sel = base_sub["priority"].values[sel_indices]
    p_v26c_sel = v26c["priority"].values[sel_indices]
    r_cons_sel = rescue_consensus[sel_indices]

    r_base_norm = np.where(p_base_sel > 0, rankdata(p_base_sel) / len(sel_indices), 0.18 * r_cons_sel)
    r_v26c_norm = np.where(p_v26c_sel > 0, rankdata(p_v26c_sel) / len(sel_indices), 0.15 * r_cons_sel)

    # 82% Abstract PRP + 18% V26C Structural Anchor
    base_rank_score = 0.82 * r_base_norm + 0.18 * r_v26c_norm

    # Gated Star-Author & Core-Topic Boost for Top 140 candidates
    gated_mask = (rankdata(-base_rank_score) <= 140).astype(np.float64)
    raw_stage2_score = base_rank_score + gated_mask * domain_boosts[sel_indices]

    # --- ENGINE 3: STRICT MONOTONIC CONVEX EXPONENTIAL MAPPING ---
    order_within_sel = np.lexsort((
        -test_df["id"].values[sel_indices],
        -r_v26c_norm,
        -r_base_norm,
        -raw_stage2_score
    ))

    final_prio = np.zeros(len(test_df), dtype=np.float64)
    M = len(sel_indices)  # 374
    for rank_0, pos_in_sel in enumerate(order_within_sel):
        global_idx = sel_indices[pos_in_sel]
        convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
        linear_step = (M - rank_0) * 1e-4
        final_prio[global_idx] = 0.050000 + 1.200000 * convex_curve + linear_step

    final_prio = np.round(final_prio, 6)
    active_final = final_prio[final_prio > 0]

    n_active = len(active_final)
    n_unique = len(np.unique(active_final))

    print("\n" + "="*75)
    print("=== [ENGINE 3: STRICT MONOTONIC CONVEX AUDIT] ===")
    print("="*75)
    print(f"Total Recommended: {n_active}/596")
    print(f"Unique Non-Zero Priorities: {n_unique}/{n_active}")
    print(f"Duplicate Priority Ties: {n_active - n_unique} (ZERO TIES GUARANTEED!)")
    assert n_active == target_budget, f"Budget mismatch: {n_active} != {target_budget}"
    assert n_unique == target_budget, f"Found {n_active - n_unique} duplicate ties!"

    print("\n" + "="*75)
    print("=== [TOP 20 DIAMOND RANKING IN V28C] ===")
    print("="*75)
    top20_idx = sel_indices[order_within_sel[:20]]
    for r_i, idx in enumerate(top20_idx, 1):
        print(f"  Rank {r_i:2d} | ID {int(test_df.loc[idx, 'id']):4d} | Prio: {final_prio[idx]:.6f} | [{venues[idx].upper():5s}] {test_df.loc[idx, 'title'][:62]}")

    out_name = "submission_v28c_374_convex_prp.csv"
    out_path = SUB_DIR / out_name
    sub_v28c = pd.DataFrame({"id": test_df["id"], "priority": final_prio})
    sub_v28c.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Saved {out_path} — Ready for Final Submission #3!")

if __name__ == "__main__":
    build_v28c_submission()
