import pandas as pd

train_df = pd.read_csv('source-code/data/train.csv')
title_lower = train_df['title'].str.lower().fillna('')
nnv_mask = title_lower.str.contains('neural network|nnv|imagestar|deep neural|verification of neural|neural networks')
nnv_papers = train_df[nnv_mask]

print("="*80)
print(f"=== NEURAL NETWORK VERIFICATION PAPERS IN TRAIN.CSV ({len(nnv_papers)} total) ===")
print("="*80)
print("Ratings distribution:")
print(nnv_papers['Label'].value_counts().sort_index())
print(f"Mean Rating: {nnv_papers['Label'].mean():.2f}")

print("\nDetails of all NNV papers:")
for _, r in nnv_papers.iterrows():
    auth = str(r['authors'])[:28] if pd.notna(r['authors']) else "Unknown"
    title = str(r['title'])[:55]
    ven = str(r['venue']).upper()
    print(f"ID {r['id']:4d} | Label: {r['Label']} | [{ven:5s}] | {auth:28s} | {title}")
