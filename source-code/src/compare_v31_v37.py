import pandas as pd

v31 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v31_grand_consensus_apex_374.csv")
v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')

v31_act = v31[v31['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

diffs = []
for i in range(len(v37_act)):
    id31 = v31_act.loc[i, 'id']
    id37 = v37_act.loc[i, 'id']
    if id31 != id37:
        diffs.append((i+1, id31, id37))

print(f"Total rank differences between V31 and V37: {len(diffs)}")
with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\v31_vs_v37_diff.txt", "w", encoding="utf-8") as f:
    f.write(f"Total differences: {len(diffs)}\n")
    f.write(f"{'Rank':4s} | {'V31 ID':6s} | {'V31 Title':35s} | {'V37 ID':6s} | {'V37 Title':35s}\n")
    f.write("-" * 95 + "\n")
    for r, id31, id37 in diffs[:40]:
        t31 = test.loc[id31]
        t37 = test.loc[id37]
        f.write(f"{r:4d} | {id31:6d} | {str(t31['title'])[:33]:35s} | {id37:6d} | {str(t37['title'])[:33]:35s}\n")
