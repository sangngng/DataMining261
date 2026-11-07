import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')
train = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\train.csv")

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\audit_tail_360_374.txt", "w", encoding="utf-8") as f:
    f.write("=== PAPERS AT RANKS 360 TO 374 IN V37 ===\n")
    for r in range(360, 375):
        pid = v37_act.loc[r-1, 'id']
        t = test.loc[pid]
        f.write(f"Rank {r:3d} | ID {pid:4d} | {t['venue']:5s} | {t['year']} | {t['title'][:45]} | {str(t['authors'])[:30]}\n")

print("Done writing audit_tail_360_374.txt")
