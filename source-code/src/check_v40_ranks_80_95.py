from test_v40_design import create_v40_candidate
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
DATA_DIR = BASE_DIR / "data"

test, v40_order = create_v40_candidate()
test = test.set_index('id')

print("=== V40 Ranks 80 to 95 ===")
for r in range(80, 96):
    pid = v40_order[r-1]
    t = test.loc[pid]
    print(f"Rank {r:3d} | ID {pid:4d} | {t['venue']:5s} | {t['year']} | {t['title'][:40]} | {str(t['authors'])[:30]}")
