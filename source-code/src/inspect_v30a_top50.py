import pandas as pd
import numpy as np

test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v30a = pd.read_csv('source-code/submissions/submission_v30a_pinnacle_apex_374.csv').set_index('id').reindex(test['id']).reset_index()
test['r30a'] = np.where(v30a['priority'] > 0, v30a['priority'].rank(ascending=False, method='min'), 999)

print("=== V30A TOP 50 RANKS ===")
top50 = test.sort_values('r30a').head(50)
for idx, r in top50.iterrows():
    t = str(r['title']).encode('ascii', 'replace').decode('ascii')
    a = str(r['authors']).encode('ascii', 'replace').decode('ascii')
    print(f"Rank {r['r30a']:2.0f} | ID {r['id']:4d} | [{r['venue'].upper():5s}] | {t[:55]} | Auth: {a[:30]}")
