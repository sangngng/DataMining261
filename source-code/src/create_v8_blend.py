import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sub_dir = os.path.join(BASE_DIR, "submissions")

v2_path = os.path.join(sub_dir, "submission_v2_sbert_ensemble.csv")
v4_path = os.path.join(sub_dir, "submission_v4_clean_ensemble.csv")

v2 = pd.read_csv(v2_path)
v4 = pd.read_csv(v4_path)

print(f"Loaded V2 (LB: 0.65006): non-zero = {(v2['priority'] > 0).sum()}/596")
print(f"Loaded V4 (LB: 0.64907): non-zero = {(v4['priority'] > 0).sum()}/596")

# Direct high-precision blend of the two best pure-semantic submissions
# Both use 100% pure semantic signals with ZERO author distortion
avg_prio = 0.50 * v2['priority'] + 0.50 * v4['priority']

# Calibrate to exactly 381 papers (the optimal count that achieved 0.65006)
sorted_idx = avg_prio.sort_values(ascending=False).index
top_381_idx = sorted_idx[:381]

v8_prio = pd.Series(0.0, index=v2.index)
v8_prio.loc[top_381_idx] = avg_prio.loc[top_381_idx]

v8_df = pd.DataFrame({
    'id': v2['id'],
    'priority': v8_prio
})

out_path = os.path.join(sub_dir, "submission_v8_pure_semantic_blend.csv")
v8_df.to_csv(out_path, index=False)

print(f"\n[SUCCESS] Generated V8 Pure Semantic Blend: {out_path}")
print(f"Non-zero count: {(v8_df['priority'] > 0).sum()} / {len(v8_df)} ({(v8_df['priority'] > 0).mean()*100:.2f}%)")
print(f"Min non-zero:   {v8_df[v8_df['priority'] > 0]['priority'].min():.4f}")
print(f"Max non-zero:   {v8_df['priority'].max():.4f}")
print(f"Mean non-zero:  {v8_df[v8_df['priority'] > 0]['priority'].mean():.4f}")
print(f"Correlation with V2: {v8_df['priority'].corr(v2['priority']):.5f}")
print(f"Correlation with V4: {v8_df['priority'].corr(v4['priority']):.5f}")
