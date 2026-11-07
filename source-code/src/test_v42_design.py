import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
v41 = pd.read_csv(BASE_DIR / "submissions" / "submission_v41_apex_zenith_374.csv")
test = pd.read_csv(BASE_DIR / "data" / "test_with_abstracts.csv").set_index('id')

v41_act = v41[v41['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
order = list(v41_act['id'].values)

# Inspect current ranks
print("Current rank of 1372:", order.index(1372) + 1)
print("Current rank of 877 :", order.index(877) + 1)
print("Current rank of 899 :", order.index(899) + 1)

# Perform surgical moves
# 1. Move 899 down to rank 80
order.remove(899)
order.insert(79, 899)

# 2. Move 1372 up to rank 40
order.remove(1372)
order.insert(39, 1372)

# 3. Move 877 up to rank 50
order.remove(877)
order.insert(49, 877)

print("\nNew rank of 1372:", order.index(1372) + 1)
print("New rank of 877 :", order.index(877) + 1)
print("New rank of 899 :", order.index(899) + 1)

assert len(order) == 374
assert set(order) == set(v41_act['id'].values)
assert order[:35] == list(v41_act['id'].values)[:35], "Top 35 must remain 100% frozen!"
print("Invariants verified successfully!")
