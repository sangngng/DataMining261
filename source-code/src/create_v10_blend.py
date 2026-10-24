import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sub_dir = os.path.join(BASE_DIR, "submissions")

v8_path = os.path.join(sub_dir, "submission_v8_pure_semantic_blend.csv")
v9_path = os.path.join(sub_dir, "submission_v9_mpnet_dense.csv")

v8 = pd.read_csv(v8_path)
v9 = pd.read_csv(v9_path)

print(f"Loaded V8 (Pure MiniLM Blend): non-zero = {(v8['priority'] > 0).sum()}/596")
print(f"Loaded V9 (MPNet-Dense Frontier): non-zero = {(v9['priority'] > 0).sum()}/596")

# Grand Master Blend: 60% V8 (MiniLM baseline anchor) + 40% V9 (MPNet 768d frontier)
blended = 0.60 * v8['priority'] + 0.40 * v9['priority']

# Select top 381 papers (golden zone)
top_381_idx = blended.sort_values(ascending=False).index[:381]

v10_prio = pd.Series(0.0, index=v8.index)
v10_prio.loc[top_381_idx] = blended.loc[top_381_idx]

v10_df = pd.DataFrame({
    'id': v8['id'],
    'priority': v10_prio
})

out_path = os.path.join(sub_dir, "submission_v10_grand_master_blend.csv")
v10_df.to_csv(out_path, index=False)

print(f"\n[SUCCESS] Generated V10 Grand Master Blend: {out_path}")
print(f"Non-zero count: {(v10_df['priority'] > 0).sum()} / {len(v10_df)} ({(v10_df['priority'] > 0).mean()*100:.2f}%)")
print(f"Min non-zero:   {v10_df[v10_df['priority'] > 0]['priority'].min():.4f}")
print(f"Max non-zero:   {v10_df['priority'].max():.4f}")
print(f"Mean non-zero:  {v10_df[v10_df['priority'] > 0]['priority'].mean():.4f}")
print(f"Correlation with V8: {v10_df['priority'].corr(v8['priority']):.5f}")
print(f"Correlation with V9: {v10_df['priority'].corr(v9['priority']):.5f}")
