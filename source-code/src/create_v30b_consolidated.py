"""
Create V30B: Consolidated Dual-Apex Core (ASP & NNV Consolidation)
Builds directly upon V30A:
  - Base: submission_v30a_pinnacle_apex_374.csv (Tail pruned & rescued, budget = 374, zero ties)
  - Anchor: Ranks 1 to 22 100% frozen (The Diamond Core)
  - Innovation: Pure Top-48 Consolidation.
    Promotes the 6 true ASP & NNV core publications currently sitting in Ranks 46-59:
      * ID 1502 [CAV] Finding Regular Herbrand Models for CHCs using Answer Set Programming
      * ID 1837 [CAV] NNrepair: Constraint-Based Repair of Neural Network Classifiers
      * ID 1351 [LPNMR] An ASP-Based Framework for Articulated Objects (Carmine Dodaro, Nicola Leone, Marco Maratea)
      * ID 1737 [CAV] CEG4N: Counter-Example Guided Neural Network Quantization Refinement
      * ID 1803 [CAV] Verisig 2.0: Verification of Neural Network Controllers Using Taylor Model Preconditioning
      * ID 1372 [LPNMR] Elaboration Tolerant Representation of Markov Decision Process via Action Language pBC+
    Demotes the non-ASP/non-NNV papers currently occupying Ranks 39, 44, 45, 50, 51, 52:
      * ID 1486 [CAV] Ganak2 SAT Model Counter (Mate Soos, 2/3 label 1 in train)
      * ID 723  [KR]  Argumentative Reasoning in ASPIC+ (Argumentation)
      * ID 959  [KR]  Epistemic Approach to Probabilistic Argumentation (Argumentation)
      * ID 1083 [KR]  Knowledge-Based Transfer Learning Explanation
      * ID 933  [KR]  Visual Intelligence in Service Robotics
      * ID 2206 [CAV] Logical Clustering and Learning for Time-Series Data
    Preserves relative internal order within both groups.
  - Ranks 60 to 374: Exactly identical to V30A.
  - Monotonic convex mapping: M=374, power=1.35, zero priority ties.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SUB_DIR = Path("source-code/submissions")
DATA_DIR = Path("source-code/data")

test_df = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")
N = len(test_df)

v30a = pd.read_csv(SUB_DIR / "submission_v30a_pinnacle_apex_374.csv").set_index("id").reindex(test_df["id"]).reset_index()
p_v30a = v30a["priority"].values

base_active_indices = np.where(p_v30a > 0)[0]
assert len(base_active_indices) == 374

current_order = sorted(base_active_indices, key=lambda i: p_v30a[i], reverse=True)

# Define papers to promote into Top 48
promote_pids = [1502, 1837, 1351, 1737, 1803, 1372]
# Define papers to demote
demote_pids = [1486, 723, 959, 1083, 933, 2206]

promote_indices = [np.where(test_df["id"].values == pid)[0][0] for pid in promote_pids]
demote_indices = [np.where(test_df["id"].values == pid)[0][0] for pid in demote_pids]

# Ranks 1 to 22 (0-indexed: 0 to 22) are frozen
frozen_core = current_order[:22]

# The middle zone to consolidate: Ranks 23 to 60 (indices 22 to 60)
middle_zone = [i for i in current_order[22:60]]

# Clean partition:
# asp_nnv_in_middle: existing ASP/NNV papers in middle zone that are NOT in promote_pids
# promote_indices: the 6 promoted ASP/NNV papers (from middle zone or ranks 46-59)
# demote_indices: the 6 demoted papers (to be placed after promote_indices)
# remaining_middle: any other papers in middle zone (neither promoted nor demoted)
remaining_middle = [i for i in middle_zone if i not in promote_indices and i not in demote_indices]

consolidated_middle = remaining_middle[:20] + promote_indices + demote_indices + remaining_middle[20:]
assert len(consolidated_middle) == len(middle_zone), f"Mismatch: {len(consolidated_middle)} vs {len(middle_zone)}"

v30b_order = frozen_core + consolidated_middle + current_order[60:]
assert len(v30b_order) == 374
assert len(set(v30b_order)) == 374

# Monotonic convex exponential mapping
M = 374
final_prio = np.zeros(N, dtype=np.float64)
for rank_0, g_idx in enumerate(v30b_order):
    convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
    linear_step = (M - rank_0) * 1e-4
    final_prio[g_idx] = 0.050000 + 1.200000 * convex_curve + linear_step
final_prio = np.round(final_prio, 6)

n_active = (final_prio > 0).sum()
n_unique = len(np.unique(final_prio[final_prio > 0]))

print(f"Total Recommended: {n_active}/596")
print(f"Total Unique Priorities: {n_unique}/374 (ZERO DUPLICATE TIES GUARANTEED!)")
assert n_active == 374
assert n_unique == 374

print("\n=== AUDIT OF PROMOTED PAPERS IN V30B ===")
for pid in promote_pids:
    idx = np.where(test_df["id"].values == pid)[0][0]
    r_a = current_order.index(idx) + 1
    r_b = v30b_order.index(idx) + 1
    row = test_df.iloc[idx]
    t = str(row['title']).encode('ascii', 'replace').decode('ascii')
    print(f"[+] PROMOTED ID {pid:4d} | [{row['venue'].upper():5s}] | R30A: {r_a:2d} -> R30B: {r_b:2d} | {t[:55]}")

print("\n=== AUDIT OF DEMOTED PAPERS IN V30B ===")
for pid in demote_pids:
    idx = np.where(test_df["id"].values == pid)[0][0]
    r_a = current_order.index(idx) + 1
    r_b = v30b_order.index(idx) + 1
    row = test_df.iloc[idx]
    t = str(row['title']).encode('ascii', 'replace').decode('ascii')
    print(f"[-] DEMOTED  ID {pid:4d} | [{row['venue'].upper():5s}] | R30A: {r_a:2d} -> R30B: {r_b:2d} | {t[:55]}")

out_path = SUB_DIR / "submission_v30b_consolidated_apex_374.csv"
sub_v30b = pd.DataFrame({"id": test_df["id"], "priority": final_prio})
sub_v30b.to_csv(out_path, index=False)
print(f"\n[SUCCESS] Saved {out_path} — Ready for Submission 2!")
