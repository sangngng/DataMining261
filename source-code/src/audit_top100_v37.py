import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
v37 = pd.read_csv(BASE_DIR / "submissions" / "submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(BASE_DIR / "data" / "test_with_abstracts.csv").set_index("id")
train = pd.read_csv(BASE_DIR / "data" / "train.csv")

# Compute author stats from train
author_stats = {}
for _, row in train.iterrows():
    label = row['Label']
    authors_str = str(row['authors'])
    if pd.isna(authors_str) or authors_str.strip() == '':
        continue
    # split authors
    for a in [x.strip().lower() for x in authors_str.split(',')]:
        if len(a) > 2:
            if a not in author_stats:
                author_stats[a] = []
            author_stats[a].append(label)

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

rows = []
for rank in range(1, 101):
    pid = v37_act.loc[rank - 1, 'id']
    t_row = test.loc[pid]
    title = str(t_row['title'])
    authors = str(t_row['authors'])
    venue = str(t_row['venue'])
    year = t_row['year']
    
    # check author overlap with train
    matched_authors = []
    if pd.notna(authors) and authors.strip() != '':
        for a in [x.strip().lower() for x in authors.split(',')]:
            for ta, labels in author_stats.items():
                if a == ta or (len(a) > 5 and a in ta) or (len(ta) > 5 and ta in a):
                    matched_authors.append((ta, len(labels), round(sum(labels)/len(labels), 2)))
                    break
    
    rows.append({
        'rank': rank,
        'id': pid,
        'venue': venue,
        'year': year,
        'title': title[:50],
        'authors': authors[:40],
        'matched_author_stats': matched_authors[:2]
    })

audit_df = pd.DataFrame(rows)
out_txt = BASE_DIR / "src" / "top100_audit.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(audit_df.to_string())
print(f"Written {len(audit_df)} rows to {out_txt}")
