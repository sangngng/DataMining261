import pandas as pd
import numpy as np
from scipy.stats import rankdata

test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v29c = pd.read_csv('source-code/submissions/submission_v29c_late_fusion_apex.csv').set_index('id').reindex(test['id']).reset_index()
v29a = pd.read_csv('source-code/submissions/submission_v29a_hyper_top25_374.csv').set_index('id').reindex(test['id']).reset_index()
v28c = pd.read_csv('source-code/submissions/submission_v28c_374_convex_prp.csv').set_index('id').reindex(test['id']).reset_index()
v26c = pd.read_csv('source-code/submissions/submission_v26c_cluster_boundary_365.csv').set_index('id').reindex(test['id']).reset_index()
v27  = pd.read_csv('source-code/submissions/submission_v27_apex_curriculum_365.csv').set_index('id').reindex(test['id']).reset_index()

p_v29c = v29c['priority'].values
p_v29a = v29a['priority'].values
p_v28c = v28c['priority'].values
p_v26c = v26c['priority'].values
p_v27  = v27['priority'].values

active_374 = np.where(p_v29c > 0)[0]
assert len(active_374) == 374

# Normalized rank distributions within active set
r29c = rankdata(p_v29c[active_374]) / 374.0
r29a = rankdata(p_v29a[active_374]) / 374.0
r28c = rankdata(p_v28c[active_374]) / 374.0

# Impute for 365-budget models using r29c
r26c_imp = np.where(p_v26c[active_374] > 0, rankdata(p_v26c[active_374]) / 374.0, r29c)
r27_imp  = np.where(p_v27[active_374] > 0,  rankdata(p_v27[active_374]) / 374.0,  r29c)

# Diamond 22 papers (Anchor)
DIAMOND_IDS = [1343, 1313, 1378, 1330, 979, 1185, 1395, 1143, 991, 767,
               1122, 1321, 1337, 1305, 1296, 1302, 911, 1329, 1920, 1648, 1835, 1921]

diamond_indices = [i for i in active_374 if test.loc[i, 'id'] in DIAMOND_IDS]
diamond_sorted = sorted(diamond_indices, key=lambda i: p_v29c[i], reverse=True)

# Test Ensemble Weights:
# Strongly anchored on V29C (45%) + V29A (25%) + V28C (15%) + V26C (10%) + V27 (5%)
blend_score = 0.45 * r29c + 0.25 * r29a + 0.15 * r28c + 0.10 * r26c_imp + 0.05 * r27_imp

other_indices = [i for i in active_374 if test.loc[i, 'id'] not in DIAMOND_IDS]
other_sorted = sorted(other_indices, key=lambda i: blend_score[np.where(active_374 == i)[0][0]], reverse=True)

final_order = diamond_sorted + other_sorted
assert len(final_order) == 374
assert set(final_order) == set(active_374)

# Convex exponential priority mapping
M = 374
final_prio = np.zeros(len(test), dtype=np.float64)
for rank_0, g_idx in enumerate(final_order):
    convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
    linear_step = (M - rank_0) * 1e-4
    final_prio[g_idx] = 0.050000 + 1.200000 * convex_curve + linear_step
final_prio = np.round(final_prio, 6)

# Check how many papers changed rank vs V29C
v29c_order = list(v29c.sort_values('priority', ascending=False).head(374).index)
shifts = [abs(final_order.index(idx) - v29c_order.index(idx)) for idx in active_374]
print(f"Max rank shift vs V29C: {max(shifts)}")
print(f"Mean rank shift vs V29C: {np.mean(shifts):.2f}")
print(f"Number of papers shifted: {sum(1 for s in shifts if s > 0)}")

# Save as candidate submission
sub_df = pd.DataFrame({'id': test['id'], 'priority': final_prio})
sub_df.to_csv('source-code/submissions/submission_v31_grand_consensus_apex_374.csv', index=False)
print("Saved source-code/submissions/submission_v31_grand_consensus_apex_374.csv successfully!")
