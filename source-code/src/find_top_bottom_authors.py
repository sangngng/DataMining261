import pandas as pd

train = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\train.csv")

author_dict = {}
for _, row in train.iterrows():
    lbl = row['Label']
    auths = str(row['authors'])
    if pd.isna(auths) or auths == 'nan':
        continue
    for a in [x.strip() for x in auths.split(',')]:
        if len(a) > 2:
            a_lower = a.lower()
            if a_lower not in author_dict:
                author_dict[a_lower] = []
            author_dict[a_lower].append(lbl)

top_authors = []
for a, lbls in author_dict.items():
    if len(lbls) >= 2:
        top_authors.append({
            'author': a,
            'count': len(lbls),
            'mean': sum(lbls) / len(lbls),
            'labels': sorted(lbls, reverse=True)
        })

df = pd.DataFrame(top_authors).sort_values(by=['mean', 'count'], ascending=[False, False])
with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\author_stats.txt", "w", encoding="utf-8") as f:
    f.write("=== Top Authors with mean >= 3.5 (count >= 2) ===\n")
    f.write(df[df['mean'] >= 3.5].to_string())
    f.write("\n\n=== Bottom Authors with mean <= 1.8 (count >= 3) ===\n")
    f.write(df[(df['mean'] <= 1.8) & (df['count'] >= 3)].to_string())
print("Success writing author_stats.txt")
