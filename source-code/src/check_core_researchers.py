import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index("id")

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v37_rank_map = {row['id']: r + 1 for r, row in v37_act.iterrows()}

core_names = [
    "schaub", "eiter", "tran cao son", "son tran", "banbara", "inoue", "cabalar",
    "gebser", "ricca", "faber", "alviano", "redl", "lierler", "poole", "fichte",
    "saribatur", "katz", "bak", "hoang-dung tran", "komendantskaya", "dwyer",
    "shriver", "daggitt", "weinzierl", "sabuncu", "formisano", "pontelli",
    "marques-silva", "ignatiev", "darwiche", "meel", "dovier", "delgrande", "kakas"
]

results = []
for pid, t_row in test.iterrows():
    auths = str(t_row['authors']).lower()
    title = str(t_row['title'])
    venue = str(t_row['venue'])
    year = t_row['year']
    rank = v37_rank_map.get(pid, 999)
    
    matched = [name for name in core_names if name in auths]
    if matched:
        results.append({
            'id': pid,
            'v37_rank': rank,
            'matched': ", ".join(matched),
            'venue': venue,
            'year': year,
            'title': title[:55],
            'authors': str(t_row['authors'])[:45]
        })

df_res = pd.DataFrame(results).sort_values(by='v37_rank')
out_txt = r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\core_researchers_test_ranks.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(f"Total core researcher papers in test: {len(df_res)}\n")
    f.write(df_res.to_string())
print(f"Total: {len(df_res)}")
