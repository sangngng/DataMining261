import pandas as pd
import numpy as np

train = pd.read_csv('source-code/data/train.csv')
test = pd.read_csv('source-code/data/test_with_abstracts.csv')

m = train['title'].fillna('').str.lower().str.contains('vehicle|traffic|autonomous vehicle|connected')
print('Matching papers in train:', m.sum())
sub = train[m][['id', 'venue', 'Label', 'title']]
for idx, r in sub.iterrows():
    t = r['title'].encode('ascii', 'replace').decode('ascii')
    print(f"{r['id']:4d} | {r['venue']:5s} | Label: {r['Label']} | {t[:60]}")
print('Mean rating:', sub['Label'].mean(), 'Pos rate:', (sub['Label']>=2).mean())
