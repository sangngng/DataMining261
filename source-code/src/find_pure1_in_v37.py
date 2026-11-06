import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index("id")
train = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\train.csv")

# Find pure label 1 authors (count >= 2, all 1s)
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

pure_1_authors = {a: len(lbls) for a, lbls in author_dict.items() if len(lbls) >= 2 and all(x == 1 for x in lbls)}

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

found_in_active = []
for rank_0, row in v37_act.iterrows():
    pid = row['id']
    t_row = test.loc[pid]
    auths = str(t_row['authors'])
    if pd.isna(auths) or auths == 'nan':
        continue
    for a in [x.strip().lower() for x in auths.split(',')]:
        if a in pure_1_authors:
            found_in_active.append({
                'rank': rank_0 + 1,
                'id': pid,
                'author': a,
                'train_1s': pure_1_authors[a],
                'title': str(t_row['title'])[:60],
                'venue': t_row['venue']
            })

df_found = pd.DataFrame(found_in_active)
print(f"Total pure label 1 author papers in active 374: {len(df_found)}")
print(df_found.to_string())
