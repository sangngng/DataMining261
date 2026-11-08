import pandas as pd
import numpy as np

test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v44 = pd.read_csv('source-code/submissions/submission_v44_apex_hyper_consensus_374.csv')
v43 = pd.read_csv('source-code/submissions/submission_v43_multi_apex_consensus_374.csv')

# Audits
assert len(v44) == 596
assert (v44['priority'] > 0).sum() == 374
assert (v44['priority'] == 0).sum() == 222
assert len(v44['priority'].unique()) == 375
assert (v44['id'] == test['id']).all()

# Check active paper ranks
act44 = v44[v44['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
act43 = v43[v43['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

test_indexed = test.set_index('id')

print("=== RANKS 1 TO 25 IN V44 ===")
for r in range(1, 26):
    pid44 = act44.loc[r-1, 'id']
    pid43 = act43.loc[r-1, 'id']
    t44 = test_indexed.loc[pid44]
    diff_mark = " (NEW/SHIFTED)" if pid44 != pid43 else ""
    print(f"Rank {r:2d} | ID {pid44:4d} | Prio {act44.loc[r-1, 'priority']:.6f} | {t44['venue']:5s} | {str(t44['title'])[:45]:47s}{diff_mark}")

print(f"\n#878 rank in V44: {act44[act44['id'] == 878].index[0] + 1}")
print(f"#878 rank in V43: {act43[act43['id'] == 878].index[0] + 1}")
