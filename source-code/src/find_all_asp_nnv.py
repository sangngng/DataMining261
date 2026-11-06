import pandas as pd
import re

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')
v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v37_rank_map = {row['id']: r + 1 for r, row in v37_act.iterrows()}

# Let's search for keywords in title and abstract
asp_pattern = re.compile(r'\b(answer set|asp|clingo|potassco|stable model|datalog|dlv)\b', re.IGNORECASE)
nnv_pattern = re.compile(r'\b(neural network verification|neural network robustness|nnv|dnn verification|marabou|reluplex|eran)\b', re.IGNORECASE)

asp_papers = []
nnv_papers = []

for pid, t in test.iterrows():
    text = f"{t['title']} {t['abstract']}"
    r = v37_rank_map.get(pid, 999)
    if asp_pattern.search(text):
        asp_papers.append((r, pid, t['venue'], t['year'], t['title'], t['authors']))
    if nnv_pattern.search(text):
        nnv_papers.append((r, pid, t['venue'], t['year'], t['title'], t['authors']))

asp_papers.sort(key=lambda x: x[0])
nnv_papers.sort(key=lambda x: x[0])

with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\asp_nnv_audit.txt", "w", encoding="utf-8") as f:
    f.write(f"=== ALL ASP PAPERS IN TEST ({len(asp_papers)}) ===\n")
    for r, pid, venue, yr, title, auth in asp_papers:
        f.write(f"Rank {r:3d} | ID {pid:4d} | {venue:5s} | {yr} | {title[:50]:50s} | {str(auth)[:35]}\n")
    f.write(f"\n=== ALL NNV PAPERS IN TEST ({len(nnv_papers)}) ===\n")
    for r, pid, venue, yr, title, auth in nnv_papers:
        f.write(f"Rank {r:3d} | ID {pid:4d} | {venue:5s} | {yr} | {title[:50]:50s} | {str(auth)[:35]}\n")

print("Done writing asp_nnv_audit.txt")
