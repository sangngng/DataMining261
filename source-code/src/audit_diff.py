import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
v37_path = BASE_DIR / "submissions" / "submission_v37_pinnacle_apex_374.csv"
v39_path = BASE_DIR / "submissions" / "submission_v39_pure_consensus_apex_374.csv"
test_path = BASE_DIR / "data" / "test_with_abstracts.csv"

v37 = pd.read_csv(v37_path)
v39 = pd.read_csv(v39_path)
test = pd.read_csv(test_path).set_index("id")

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v39_act = v39[v39['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

diffs = []
for i in range(len(v37_act)):
    id37 = v37_act.loc[i, 'id']
    id39 = v39_act.loc[i, 'id']
    if id37 != id39:
        diffs.append((i+1, id37, id39))

print(f"Total rank differences: {len(diffs)}")
print(f"{'Rank':4s} | {'V37 ID':6s} | {'V39 ID':6s} | {'V37 Title':40s} | {'V39 Title':40s}")
print("-" * 105)
for r, id37, id39 in diffs:
    t37 = str(test.loc[id37, 'title'])[:38] if id37 in test.index else "N/A"
    t39 = str(test.loc[id39, 'title'])[:38] if id39 in test.index else "N/A"
    print(f"{r:4d} | {id37:6d} | {id39:6d} | {t37:40s} | {t39:40s}")
