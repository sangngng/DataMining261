import pandas as pd
import numpy as np

train_df = pd.read_csv('source-code/data/train.csv')
p1 = train_df[train_df['id'] < 713]
p2 = train_df[train_df['id'] >= 713]

print("="*80)
print("=== PART 1 vs PART 2 LABEL DISTRIBUTIONS ===")
print("="*80)
print(f"Part 1: Total={len(p1)}, Positive Rate={ (p1['Label']>=2).mean():.4f}")
print(p1['Label'].value_counts().sort_index())

print(f"\nPart 2: Total={len(p2)}, Positive Rate={ (p2['Label']>=2).mean():.4f}")
print(p2['Label'].value_counts().sort_index())

print("\n" + "="*80)
print("=== PART 2 BREAKDOWN BY VENUE ===")
print("="*80)
for v, grp in p2.groupby('venue'):
    pos_rate = (grp['Label'] >= 2).mean()
    rate_5 = (grp['Label'] == 5).mean()
    rate_4 = (grp['Label'] == 4).mean()
    rate_3 = (grp['Label'] == 3).mean()
    rate_2 = (grp['Label'] == 2).mean()
    rate_1 = (grp['Label'] == 1).mean()
    mean_rating = grp['Label'].mean()
    print(f"Venue {v.upper():6s} | N={len(grp):4d} | Mean={mean_rating:.2f} | Pos={pos_rate:.3f} | 5*={rate_5:.3f} | 4*={rate_4:.3f} | 3*={rate_3:.3f} | 2*={rate_2:.3f} | 1*={rate_1:.3f}")

test_df = pd.read_csv('source-code/data/test_with_abstracts.csv')
print("\n" + "="*80)
print("=== EXPECTED LABELS IN TEST SET (596 items, all in Part 2) ===")
print("="*80)
expected_pos = 0.0
expected_5 = 0.0
expected_4 = 0.0
for v, grp in test_df.groupby('venue'):
    p2_ven = p2[p2['venue'] == v]
    pos_r = (p2_ven['Label'] >= 2).mean()
    r5 = (p2_ven['Label'] == 5).mean()
    r4 = (p2_ven['Label'] == 4).mean()
    n_v = len(grp)
    exp_pos_v = n_v * pos_r
    exp_5_v = n_v * r5
    exp_4_v = n_v * r4
    expected_pos += exp_pos_v
    expected_5 += exp_5_v
    expected_4 += exp_4_v
    print(f"Test Venue {v.upper():6s} | Total in Test: {n_v:3d} | Exp Positives: {exp_pos_v:5.1f} | Exp 5*: {exp_5_v:4.1f} | Exp 4*: {exp_4_v:4.1f}")

print(f"\nTOTAL EXPECTED POSITIVES IN TEST SET: {expected_pos:.1f} / 596 ({expected_pos/596*100:.1f}%)")
print(f"TOTAL EXPECTED 5-STAR PAPERS IN TEST SET: {expected_5:.1f}")
print(f"TOTAL EXPECTED 4-STAR PAPERS IN TEST SET: {expected_4:.1f}")
print(f"TOTAL EXPECTED (4* + 5*) IN TEST SET: {expected_5 + expected_4:.1f}")
