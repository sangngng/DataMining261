import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
train_path = BASE_DIR / "data" / "train.csv"
train = pd.read_csv(train_path)

print(f"Total train papers: {len(train)}")
print("Label distribution in train:")
vc = train['Label'].value_counts().sort_index()
for rating, count in vc.items():
    print(f"  Rating {rating}: {count} ({count/len(train)*100:.2f}%)")

worthwhile = (train['Label'] > 1).sum()
print(f"Total worthwhile (Label > 1): {worthwhile} ({worthwhile/len(train)*100:.2f}%)")
print(f"Total unworthwhile (Label == 1): {len(train) - worthwhile} ({(len(train)-worthwhile)/len(train)*100:.2f}%)")
