import pandas as pd

test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v46 = pd.read_csv('source-code/submissions/submission_v46_apex_transcendence_374.csv')
v45 = pd.read_csv('source-code/submissions/submission_v45_apex_triad_consensus_374.csv')

assert len(v46) == 596
assert (v46['id'] == test['id']).all()
assert (v46['priority'] > 0).sum() == 374
assert (v46['priority'] == 0).sum() == 222
assert len(v46['priority'].unique()) == 375

act46 = v46[v46['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
act45 = v45[v45['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

test_idx = test.set_index('id')

print("=== V46 AUDIT REPORT ===")
print(f"Top 25 Identical to V45: {list(act46['id'][:25]) == list(act45['id'][:25])}")
print(f"#878 Rank:  {act46[act46['id'] == 878].index[0] + 1}")
print(f"#959 Rank:  {act46[act46['id'] == 959].index[0] + 1}")
print(f"#1486 Rank: {act46[act46['id'] == 1486].index[0] + 1}")
print(f"#1053 Rank: {act46[act46['id'] == 1053].index[0] + 1}")
print(f"#1264 Rank: {act46[act46['id'] == 1264].index[0] + 1}")
print(f"#1419 Rank: {act46[act46['id'] == 1419].index[0] + 1}")

diff_count = (act46['id'] != act45['id']).sum()
print(f"Total Rank Differences vs V45: {diff_count}")

print("\n--- RANKS 50 TO 60 IN V46 ---")
for r in range(50, 61):
    pid = act46.loc[r - 1, 'id']
    t = test_idx.loc[pid]
    print(f"Rank {r:2d} | ID {pid:4d} | Prio {act46.loc[r-1, 'priority']:.6f} | {t['venue']:5s} | {str(t['title'])[:45]}")
