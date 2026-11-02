import pandas as pd
import numpy as np

sub_a = pd.read_csv('source-code/submissions/submission_v29a_hyper_top25_374.csv').set_index('id')
sub_b = pd.read_csv('source-code/submissions/submission_v29b_budget374_edge_rescue.csv').set_index('id')
test_df = pd.read_csv('source-code/data/test_with_abstracts.csv')

sub_a['rank_a'] = np.where(sub_a['priority'] > 0, sub_a['priority'].rank(ascending=False), 999)
sub_b['rank_b'] = np.where(sub_b['priority'] > 0, sub_b['priority'].rank(ascending=False), 999)

test_df['ra'] = sub_a.loc[test_df['id'], 'rank_a'].values
test_df['rb'] = sub_b.loc[test_df['id'], 'rank_b'].values
test_df['delta_rank'] = test_df['ra'] - test_df['rb'] # positive means promoted in B

print("="*85)
print("=== PAPERS PROMOTED IN V29B ===")
print("="*85)
promoted = test_df.sort_values('delta_rank', ascending=False).head(12)
for _, r in promoted.iterrows():
    auth = str(r['authors'])[:25] if pd.notna(r['authors']) else "Unknown"
    title = str(r['title'])[:45]
    ven = str(r['venue']).upper()
    print(f"ID {r['id']:4d} | [{ven:5s}] | Rank A: {r['ra']:3.0f} -> Rank B: {r['rb']:3.0f} (diff={r['delta_rank']:+.0f}) | {auth:25s} | {title}")

print("\n" + "="*85)
print("=== PAPERS DEMOTED IN V29B ===")
print("="*85)
demoted = test_df.sort_values('delta_rank', ascending=True).head(12)
for _, r in demoted.iterrows():
    auth = str(r['authors'])[:25] if pd.notna(r['authors']) else "Unknown"
    title = str(r['title'])[:45]
    ven = str(r['venue']).upper()
    print(f"ID {r['id']:4d} | [{ven:5s}] | Rank A: {r['ra']:3.0f} -> Rank B: {r['rb']:3.0f} (diff={r['delta_rank']:+.0f}) | {auth:25s} | {title}")

print("\n" + "="*85)
print("=== EXACT RANKS 23 TO 35 IN V29A vs V29B ===")
print("="*85)
print("V29A Ranks 23 to 35:")
for _, r in test_df[(test_df['ra'] >= 23) & (test_df['ra'] <= 35)].sort_values('ra').iterrows():
    auth = str(r['authors'])[:25] if pd.notna(r['authors']) else "Unknown"
    title = str(r['title'])[:45]
    ven = str(r['venue']).upper()
    print(f"  Rank {r['ra']:2.0f} in A -> Rank {r['rb']:2.0f} in B | ID {r['id']:4d} | [{ven:5s}] | {auth:25s} | {title}")

print("\nV29B Ranks 23 to 35:")
for _, r in test_df[(test_df['rb'] >= 23) & (test_df['rb'] <= 35)].sort_values('rb').iterrows():
    auth = str(r['authors'])[:25] if pd.notna(r['authors']) else "Unknown"
    title = str(r['title'])[:45]
    ven = str(r['venue']).upper()
    print(f"  Rank {r['rb']:2.0f} in B <- Rank {r['ra']:2.0f} in A | ID {r['id']:4d} | [{ven:5s}] | {auth:25s} | {title}")
