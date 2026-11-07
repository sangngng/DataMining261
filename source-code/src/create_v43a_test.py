import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

test = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")
v41 = pd.read_csv(SUB_DIR / "submission_v41_apex_zenith_374.csv").set_index('id').reindex(test['id']).reset_index()

# 1. Candidate V43A: Tail Trim M=372 (Cuts 2 lowest tail noise papers from V41)
v41_act = v41[v41['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
order_372 = list(v41_act['id'].values)[:372]
cut_ids = list(v41_act['id'].values)[372:]
print(f"Cut papers from tail (Ranks 373-374): {cut_ids}")
for cid in cut_ids:
    t = test[test['id'] == cid].iloc[0]
    print(f"  ID {cid:4d} | {t['venue']} {t['year']} | {t['title'][:55]} | {str(t['authors'])[:30]}")

def save_sub(active_order, M, out_path):
    final_prio = np.zeros(len(test), dtype=np.float64)
    for rank_0, p_id in enumerate(active_order):
        g_idx = test[test["id"] == p_id].index[0]
        convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
        linear_step = (M - rank_0) * 1e-4
        final_prio[g_idx] = 0.050000 + 1.200000 * convex_curve + linear_step
    final_prio = np.round(final_prio, 6)
    sub_df = pd.DataFrame({'id': test['id'], 'priority': final_prio})
    assert len(sub_df) == 596
    assert (sub_df['priority'] > 0).sum() == M
    assert len(sub_df['priority'].unique()) == M + 1
    sub_df.to_csv(out_path, index=False)
    print(f"Generated {out_path} (Active={M}, Zero ties guaranteed!)")

save_sub(order_372, 372, SUB_DIR / "submission_v43a_tail_trim_372.csv")
