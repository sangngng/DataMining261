import pandas as pd
import numpy as np

sub_v28c = pd.read_csv('source-code/submissions/submission_v28c_374_convex_prp.csv')
test_df = pd.read_csv('source-code/data/test_with_abstracts.csv')
merged = pd.merge(test_df, sub_v28c, on='id')
top30 = merged.sort_values('priority', ascending=False).head(30)

print("="*85)
print("=== TOP 30 PAPERS IN V28C CURRENTLY ===")
print("="*85)
for r, (_, row) in enumerate(top30.iterrows(), 1):
    auth = str(row['authors'])[:32] if pd.notna(row['authors']) else "Unknown"
    title = str(row['title'])[:52]
    venue = str(row['venue']).upper()
    print(f"Rank {r:2d} | ID {row['id']:4d} | Prio: {row['priority']:.6f} | [{venue:5s}] | {auth:32s} | {title}")

sub_v26c = pd.read_csv('source-code/submissions/submission_v26c_cluster_boundary_365.csv').set_index('id')
sub_v25b = pd.read_csv('source-code/submissions/submission_v25b_transformer_hybrid_365.csv').set_index('id')
sub_v20 = pd.read_csv('source-code/submissions/submission_v20_grand_apex.csv').set_index('id')
sub_v27 = pd.read_csv('source-code/submissions/submission_v27_apex_curriculum_365.csv').set_index('id')
sub_v28c_idx = sub_v28c.set_index('id')
v28c_zeros = test_df[sub_v28c_idx.loc[test_df['id'], 'priority'].values == 0].copy()
v28c_zeros['v27'] = sub_v27.loc[v28c_zeros['id'], 'priority'].values
v28c_zeros['v26c'] = sub_v26c.loc[v28c_zeros['id'], 'priority'].values
v28c_zeros['v25b'] = sub_v25b.loc[v28c_zeros['id'], 'priority'].values
v28c_zeros['v20'] = sub_v20.loc[v28c_zeros['id'], 'priority'].values

positive_in_any = v28c_zeros[(v28c_zeros['v27'] > 0) | (v28c_zeros['v26c'] > 0) | (v28c_zeros['v25b'] > 0) | (v28c_zeros['v20'] > 0)]
print("\n" + "="*85)
print(f"=== PAPERS ZERO IN V28C BUT POSITIVE IN PRIOR TOP MODELS ({len(positive_in_any)} total) ===")
merged = pd.merge(test_df, sub_v28c, on='id').sort_values('priority', ascending=False).reset_index(drop=True)
merged['rank'] = merged.index + 1
active = merged[merged['priority'] > 0]
tail25 = active.tail(25)

print("\n" + "="*85)
print("=== BOTTOM 25 PAPERS IN V28C (RANKS 350-374) ===")
print("="*85)
for _, row in tail25.iterrows():
    auth = str(row['authors'])[:25] if pd.notna(row['authors']) else 'Unknown'
    title = str(row['title'])[:55]
    ven = str(row['venue']).upper()
    print(f"Rank {row['rank']:3d} | ID {row['id']:4d} | Prio: {row['priority']:.5f} | [{ven:5s}] | {auth:25s} | {title}")



