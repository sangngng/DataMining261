import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index("id")
train = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\train.csv")

# Find top authors in train (mean >= 4.0, count >= 2)
author_dict = {}
for _, row in train.iterrows():
    lbl = row['Label']
    auths = str(row['authors'])
    if pd.isna(auths) or auths == 'nan':
        continue
    for a in [x.strip().lower() for x in auths.split(',')]:
        if len(a) > 2:
            if a not in author_dict:
                author_dict[a] = []
            author_dict[a].append(lbl)

top_authors = {a: (len(lbls), sum(lbls)/len(lbls)) for a, lbls in author_dict.items() if len(lbls) >= 2 and (sum(lbls)/len(lbls)) >= 4.0}

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v37_rank_map = {row['id']: r + 1 for r, row in v37_act.iterrows()}

found_papers = []
for pid, t_row in test.iterrows():
    auths = str(t_row['authors'])
    if pd.isna(auths) or auths == 'nan':
        continue
    current_rank = v37_rank_map.get(pid, 999)
    for a in [x.strip().lower() for x in auths.split(',')]:
        if a in top_authors:
            cnt, mean_score = top_authors[a]
            found_papers.append({
                'id': pid,
                'v37_rank': current_rank,
                'author': a,
                'cnt': cnt,
                'mean': round(mean_score, 2),
                'venue': t_row['venue'],
                'year': t_row['year'],
                'title': str(t_row['title'])[:60]
            })

df_top = pd.DataFrame(found_papers).sort_values(by=['v37_rank'], ascending=True).drop_duplicates(subset=['id'])
out_txt = r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\top_author_papers_in_test.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(df_top.to_string())
print(f"Total test papers by top authors: {len(df_top)}")
