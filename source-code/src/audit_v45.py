import pandas as pd

test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v45 = pd.read_csv('source-code/submissions/submission_v45_apex_triad_consensus_374.csv')
v44 = pd.read_csv('source-code/submissions/submission_v44_apex_hyper_consensus_374.csv')

assert len(v45) == 596
assert (v45['id'] == test['id']).all()
assert (v45['priority'] > 0).sum() == 374
assert (v45['priority'] == 0).sum() == 222
assert len(v45['priority'].unique()) == 375

act45 = v45[v45['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
act44 = v44[v44['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

test_idx = test.set_index('id')

print("=== V45 AUDIT PASSED 100% ===")
print(f"Top 16 Identical to V44: {list(act45['id'][:16]) == list(act44['id'][:16])}")
print(f"#878 Rank: {act45[act45['id'] == 878].index[0] + 1}")
print(f"#1393 Rank: {act45[act45['id'] == 1393].index[0] + 1}")
print(f"#1877 Rank: {act45[act45['id'] == 1877].index[0] + 1}")
print(f"#1302 Rank: {act45[act45['id'] == 1302].index[0] + 1}")
print(f"#1405 Rank: {act45[act45['id'] == 1405].index[0] + 1}")
print(f"#1003 Rank: {act45[act45['id'] == 1003].index[0] + 1}")
print(f"#1342 Rank: {act45[act45['id'] == 1342].index[0] + 1}")
print(f"#1053 Rank: {act45[act45['id'] == 1053].index[0] + 1}")

diff_count = (act45['id'] != act44['id']).sum()
print(f"Total Rank Differences vs V44: {diff_count}")
