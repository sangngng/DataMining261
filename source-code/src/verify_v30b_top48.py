import pandas as pd
test = pd.read_csv('source-code/data/test_with_abstracts.csv')
v30b = pd.read_csv('source-code/submissions/submission_v30b_consolidated_apex_374.csv').set_index('id').reindex(test['id']).reset_index()
test['r30b'] = v30b['priority'].rank(ascending=False, method='min')

top48 = test.sort_values('r30b').head(48)
print('=== TOP 48 IN V30B ===')
for idx, r in top48.iterrows():
    t = str(r['title']).encode('ascii', 'replace').decode('ascii')
    print(f"Rank {r['r30b']:2.0f} | ID {r['id']:4d} | [{r['venue'].upper():5s}] | {t[:60]}")
