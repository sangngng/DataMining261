import pandas as pd
import numpy as np
from scipy.stats import rankdata
from pathlib import Path

SUB_DIR = Path("source-code/submissions")
DATA_DIR = Path("source-code/data")

test_df = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")
N = len(test_df)

v31 = pd.read_csv(SUB_DIR / "submission_v31_grand_consensus_apex_374.csv").set_index("id").reindex(test_df["id"]).reset_index()
v29c = pd.read_csv(SUB_DIR / "submission_v29c_late_fusion_apex.csv").set_index("id").reindex(test_df["id"]).reset_index()

p_v31 = v31["priority"].values
p_v29c = v29c["priority"].values

active_374 = np.where(p_v31 > 0)[0]
assert len(active_374) == 374

# High-confidence positive papers in the 222 pruned set discovered via author Bayesian history:
# ID 888  [KR]   Jean Christoph Jung (n=7, mean=3.43)
# ID 2639 [LICS] Jean Christoph Jung (n=7, mean=3.43)
# ID 2432 [LICS] Egor V. Kostylev (n=2, mean=4.00)
# ID 1859 [CAV]  Joost-Pieter Katoen (n=9, mean=3.33)
# ID 2102 [CAV]  Christopher Hahn (n=4, mean=3.25)
# ID 2507 [LICS] Tobias Winkler (n=3, mean=4.00)
# ID 2098 [CAV]  Chuchu Fan (n=2, mean=4.00)
# ID 1990 [CAV]  Mingshuai Chen (n=3, mean=3.67)

gem_pids = [888, 2639, 2432, 1859, 2102, 2507, 2098, 1990]
gem_indices = [np.where(test_df["id"].values == pid)[0][0] for pid in gem_pids]

for g_idx in gem_indices:
    assert g_idx not in active_374, f"ID {test_df.loc[g_idx, 'id']} is already active!"

print("=== 8 HIGH-VALUE GEMS IN PRUNED SET ===")
for g_idx in gem_indices:
    row = test_df.iloc[g_idx]
    print(f"ID {row.id:4d} | [{row.venue.upper():5s}] | {row.title[:65]}")

# Model 1: V32A - Budget 376 (Rescue top 2 gems: 888 and 2432)
# Preserves 100% of V31's 374 papers, adds 2 gems at Rank ~360 (safely above bottom)
order_v31 = sorted(active_374, key=lambda i: p_v31[i], reverse=True)

# Insert 2 gems (888 and 2432) at position 350
top_gems_2 = [np.where(test_df["id"].values == pid)[0][0] for pid in [888, 2432]]
order_376 = order_v31[:350] + top_gems_2 + order_v31[350:]
assert len(order_376) == 376

M = 376
prio_376 = np.zeros(N, dtype=np.float64)
for rank_0, g_idx in enumerate(order_376):
    convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
    linear_step = (M - rank_0) * 1e-4
    prio_376[g_idx] = 0.050000 + 1.200000 * convex_curve + linear_step
prio_376 = np.round(prio_376, 6)

sub_376 = pd.DataFrame({"id": test_df["id"], "priority": prio_376})
sub_376.to_csv(SUB_DIR / "submission_v32a_frontier_expansion_376.csv", index=False)
print("\n[SUCCESS] Generated submission_v32a_frontier_expansion_376.csv (Budget: 376, Zero Ties)")

# Model 2: V32B - Budget 378 (Rescue top 4 gems: 888, 2432, 1859, 2102)
top_gems_4 = [np.where(test_df["id"].values == pid)[0][0] for pid in [888, 2432, 1859, 2102]]
order_378 = order_v31[:350] + top_gems_4 + order_v31[350:]
assert len(order_378) == 378

M = 378
prio_378 = np.zeros(N, dtype=np.float64)
for rank_0, g_idx in enumerate(order_378):
    convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
    linear_step = (M - rank_0) * 1e-4
    prio_378[g_idx] = 0.050000 + 1.200000 * convex_curve + linear_step
prio_378 = np.round(prio_378, 6)

sub_378 = pd.DataFrame({"id": test_df["id"], "priority": prio_378})
sub_378.to_csv(SUB_DIR / "submission_v32b_frontier_expansion_378.csv", index=False)
print("[SUCCESS] Generated submission_v32b_frontier_expansion_378.csv (Budget: 378, Zero Ties)")
