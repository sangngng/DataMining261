import pandas as pd
import numpy as np
from scipy.stats import rankdata
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

test = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")

v41 = pd.read_csv(SUB_DIR / "submission_v41_apex_zenith_374.csv").set_index('id').reindex(test['id'])['priority'].values
v37 = pd.read_csv(SUB_DIR / "submission_v37_pinnacle_apex_374.csv").set_index('id').reindex(test['id'])['priority'].values
v36 = pd.read_csv(SUB_DIR / "submission_v36_apex_precision_374.csv").set_index('id').reindex(test['id'])['priority'].values
v34 = pd.read_csv(SUB_DIR / "submission_v34_pure_deep_apex_374.csv").set_index('id').reindex(test['id'])['priority'].values

active_mask = (v41 > 0)
assert active_mask.sum() == 374

r41 = rankdata(v41[active_mask]) / 374.0
r37 = rankdata(v37[active_mask]) / 374.0
r36 = rankdata(v36[active_mask]) / 374.0
r34 = rankdata(v34[active_mask]) / 374.0

# Multi-Apex Consensus Blend (Strongly anchored on the dual 0.66852 peaks)
blend = 0.40 * r41 + 0.30 * r37 + 0.20 * r36 + 0.10 * r34

active_ids = test.loc[active_mask, 'id'].values
sorted_indices = np.argsort(-blend)
consensus_order = active_ids[sorted_indices]

# Compare consensus order with V41
v41_df = pd.read_csv(SUB_DIR / "submission_v41_apex_zenith_374.csv")
v41_order = v41_df[v41_df['priority'] > 0].sort_values('priority', ascending=False)['id'].values

diffs = [i for i in range(374) if consensus_order[i] != v41_order[i]]
print(f"Number of rank differences between V41 and Multi-Apex Consensus: {len(diffs)}")

print("=== TOP 30 IN MULTI-APEX CONSENSUS ===")
for r, pid in enumerate(consensus_order[:30], 1):
    t = test[test['id'] == pid].iloc[0]
    print(f"Rank {r:2d} | ID {pid:4d} | {t['venue']:5s} | {t['title'][:50]}")
