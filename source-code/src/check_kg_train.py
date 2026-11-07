import pandas as pd

train = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\train.csv")
kg = train[train['title'].str.contains('knowledge graph|kg', case=False, na=False)]
print(f"Total Knowledge Graphs papers in train: {len(kg)}")
for _, r in kg.iterrows():
    print(f"ID {r['id']} | Venue: {r['venue']} | Label: {r['Label']} | Title: {r['title']}")
print(f"Mean label of Knowledge Graphs papers: {kg['Label'].mean():.2f}")
print("Label counts:", dict(kg['Label'].value_counts()))
