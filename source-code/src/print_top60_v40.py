import pandas as pd
from test_v40_design import create_v40_candidate
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

test, v40_order = create_v40_candidate()
test = test.set_index('id')

v37 = pd.read_csv(SUB_DIR / "submission_v37_pinnacle_apex_374.csv").set_index('id').reindex(test.index).reset_index()
v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v37_order = list(v37_act['id'].values)

with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\v40_vs_v37_top60.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Rank':4s} | {'V37 ID':6s} | {'V37 Title':35s} | {'V40 ID':6s} | {'V40 Title':35s}\n")
    f.write("-" * 95 + "\n")
    for r in range(1, 61):
        id37 = v37_order[r-1]
        id40 = v40_order[r-1]
        t37 = test.loc[id37]
        t40 = test.loc[id40]
        mark = " *" if id37 != id40 else ""
        f.write(f"{r:4d} | {id37:6d} | {str(t37['title'])[:33]:35s} | {id40:6d} | {str(t40['title'])[:33]:35s}{mark}\n")

print("Done writing v40_vs_v37_top60.txt")
