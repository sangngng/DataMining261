import pandas as pd
import numpy as np

test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v28c = pd.read_csv('source-code/submissions/submission_v28c_374_convex_prp.csv').set_index('id').reindex(test['id']).reset_index()
v29a = pd.read_csv('source-code/submissions/submission_v29a_hyper_top25_374.csv').set_index('id').reindex(test['id']).reset_index()
v29b = pd.read_csv('source-code/submissions/submission_v29b_budget374_edge_rescue.csv').set_index('id').reindex(test['id']).reset_index()
v29c = pd.read_csv('source-code/submissions/submission_v29c_late_fusion_apex.csv').set_index('id').reindex(test['id']).reset_index()
v30a = pd.read_csv('source-code/submissions/submission_v30a_pinnacle_apex_374.csv').set_index('id').reindex(test['id']).reset_index()

test['r28c'] = np.where(v28c['priority'] > 0, v28c['priority'].rank(ascending=False, method='min'), 999)
test['r29a'] = np.where(v29a['priority'] > 0, v29a['priority'].rank(ascending=False, method='min'), 999)
test['r29b'] = np.where(v29b['priority'] > 0, v29b['priority'].rank(ascending=False, method='min'), 999)
test['r29c'] = np.where(v29c['priority'] > 0, v29c['priority'].rank(ascending=False, method='min'), 999)
test['r30a'] = np.where(v30a['priority'] > 0, v30a['priority'].rank(ascending=False, method='min'), 999)

print("=== CORRELATION BETWEEN RECENT SUBMISSIONS (ACTIVE PRIORITIES) ===")
df_corr = pd.DataFrame({
    'v28c': v28c['priority'],
    'v29a': v29a['priority'],
    'v29b': v29b['priority'],
    'v29c': v29c['priority'],
    'v30a': v30a['priority']
})
print(df_corr.corr())

# Check differences between V29C and V30A
diff = np.abs(test['r29c'] - test['r30a'])
print(f"\nNumber of papers with different ranks between V29C and V30A: {(diff > 0).sum()}")
changed = test[diff > 0].sort_values('r30a')
print(f"Changed papers count: {len(changed)}")
for idx, r in changed.iterrows():
    t = str(r['title']).encode('ascii', 'replace').decode('ascii')
    print(f"ID {r['id']:4d} [{r['venue'].upper():5s}] | R29C: {r['r29c']:5.1f} -> R30A: {r['r30a']:5.1f} | {t[:60]}")
