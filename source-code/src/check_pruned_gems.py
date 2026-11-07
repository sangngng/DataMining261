import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')

pruned_ids = v37[v37['priority'] == 0]['id'].values
print(f"Total pruned papers: {len(pruned_ids)}")

core_keywords = [
    "schaub", "eiter", "tran cao son", "son tran", "banbara", "inoue", "cabalar",
    "gebser", "ricca", "faber", "alviano", "redl", "lierler", "poole", "fichte",
    "saribatur", "katz", "bak", "hoang-dung tran", "komendantskaya", "dwyer",
    "shriver", "daggitt", "weinzierl", "sabuncu", "formisano", "pontelli",
    "marques-silva", "ignatiev", "darwiche", "meel", "dovier", "delgrande", "kakas",
    "answer set", "clingo", "potassco", "stable model", "dlv"
]

found = []
for pid in pruned_ids:
    t = test.loc[pid]
    txt = f"{t['title']} {t['authors']} {t['abstract']}".lower()
    matches = [k for k in core_keywords if k in txt]
    if matches:
        found.append({
            'id': pid,
            'venue': t['venue'],
            'year': t['year'],
            'title': t['title'],
            'authors': str(t['authors']),
            'matches': ", ".join(matches)
        })

df_found = pd.DataFrame(found)
print(f"Found {len(df_found)} core matches in pruned papers:")
with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\core_in_pruned.txt", "w", encoding="utf-8") as f:
    f.write(df_found.to_string())

for _, r in df_found.iterrows():
    print(f"ID {r['id']} | {r['venue']} {r['year']} | Matches: {r['matches']} | {r['title'][:50]} | {r['authors'][:30]}")
