"""EDA numbers for pipeline B (python src/eda.py > reports/eda_b.txt)."""
import pandas as pd, numpy as np, collections, re
tr=pd.read_csv('work/train.csv'); te=pd.read_csv('work/test.csv')
P=print
P('labels', tr.Label.value_counts().sort_index().to_dict(), 'share y=1 %.3f'%(tr.Label.eq(1).mean()))
P('missing train', tr.isna().sum().to_dict()); P('missing test', te.isna().sum().to_dict())
P('dup titles train', tr.title.str.lower().duplicated().sum(), 'train-test title overlap', te.title.str.lower().isin(tr.title.str.lower()).sum())
d=tr[tr.doi.duplicated(keep=False)].sort_values('doi'); P('dup DOI rows', len(d), 'label disagreement groups', (d.groupby('doi').Label.nunique()>1).sum())
P(pd.crosstab(tr.venue,tr.Label,margins=True)); P('test venues', te.venue.value_counts().to_dict())
P('label by year (mean)', tr.groupby('year').Label.mean().round(2).to_dict())
P('authors missing -> label mean', tr.groupby(tr.authors.isna()).Label.mean().round(2).to_dict())
tr['tl']=tr.title.str.split().str.len(); P('title len corr', round(np.corrcoef(tr.tl, tr.Label)[0,1],3))
pro=tr.title.str.contains(r'Proceedings|Conference|Workshop|Volume|Part [IVX]', regex=True); P('proceedings-like', pro.sum(), 'mean label', round(tr.Label[pro].mean(),2))
inv=tr.title.str.contains(r'Invited|Abstract|Keynote|Tutorial', case=False, regex=True); P('invited/abstract', inv.sum(), round(tr.Label[inv].mean(),2))
w=collections.Counter(); wy=collections.defaultdict(list)
for t,l in zip(tr.title.str.lower(),tr.Label):
    for tok in set(re.findall(r'[a-z][a-z\-]{2,}',t)): w[tok]+=1; wy[tok].append(l)
s=sorted([(np.mean(wy[k]),k,c) for k,c in w.items() if c>=25])
P('lowest words', [(k,round(m,2),c) for m,k,c in s[:12]]); P('highest words', [(k,round(m,2),c) for m,k,c in s[-12:]])
