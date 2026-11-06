import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')
v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\ranks_45_75_details.txt", "w", encoding="utf-8") as f:
    for r in range(45, 76):
        pid = v37_act.loc[r-1, 'id']
        t = test.loc[pid]
        f.write(f"--- Rank {r} | ID {pid} | {t['venue']} {t['year']} ---\n")
        f.write(f"Title: {t['title']}\n")
        f.write(f"Authors: {t['authors']}\n")
        f.write(f"Abstract: {str(t['abstract'])[:200]}...\n\n")

print("Done writing ranks_45_75_details.txt")
