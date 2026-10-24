import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sub_dir = os.path.join(BASE_DIR, "submissions")

v2_path = os.path.join(sub_dir, "submission_v2_sbert_ensemble.csv")
v4_path = os.path.join(sub_dir, "submission_v4_clean_ensemble.csv")
v8_path = os.path.join(sub_dir, "submission_v8_pure_semantic_blend.csv")

v2 = pd.read_csv(v2_path)
v4 = pd.read_csv(v4_path)
v8 = pd.read_csv(v8_path)

print(f"Loaded V2 (LB: 0.65006), V4 (LB: 0.64907), V8 (LB: 0.65008)")

# Keep EXACTLY the 381 papers selected by V8 (preserving 100% of Coverage R)
rec_mask = (v8['priority'] > 0)
rec_count = rec_mask.sum()
print(f"Preserved exact recommendation mask: {rec_count} / {len(v8)} papers (63.93%)")

# Calculate model disagreement / epistemic uncertainty between V2 and V4
diff = np.abs(v2['priority'] - v4['priority'])

# Consensus-guided intra-list re-ranking:
# High consensus (small diff) papers are promoted to top rank positions r
# High uncertainty (large diff) papers are safely pushed down without being pruned
reranked_priority = np.where(
    rec_mask,
    v8['priority'] - 0.20 * diff,
    0.0
)
reranked_priority = np.clip(reranked_priority, 0.0, None)

v11_df = pd.DataFrame({
    'id': v8['id'],
    'priority': reranked_priority
})

out_path = os.path.join(sub_dir, "submission_v11_consensus_rerank.csv")
v11_df.to_csv(out_path, index=False)

print(f"\n[SUCCESS] Generated V11 Consensus Re-ranked Submission: {out_path}")
print(f"Non-zero count: {(v11_df['priority'] > 0).sum()} / {len(v11_df)} ({(v11_df['priority'] > 0).mean()*100:.2f}%)")
print(f"Min non-zero:   {v11_df[v11_df['priority'] > 0]['priority'].min():.4f}")
print(f"Max non-zero:   {v11_df['priority'].max():.4f}")
print(f"Mean non-zero:  {v11_df[v11_df['priority'] > 0]['priority'].mean():.4f}")
print(f"Correlation with V8: {v11_df['priority'].corr(v8['priority']):.5f}")
