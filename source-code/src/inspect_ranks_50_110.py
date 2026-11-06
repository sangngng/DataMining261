import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')
v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\ranks_50_110.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Rank':4s} | {'ID':6s} | {'Venue':5s} | {'Year':4s} | {'Title':45s} | {'Authors':35s}\n")
    f.write("-" * 110 + "\n")
    for r in range(50, 110):
        pid = v37_act.loc[r-1, 'id']
        t = test.loc[pid]
        f.write(f"{r:4d} | {pid:6d} | {t['venue']:5s} | {str(t['year'])[:4]:4s} | {str(t['title'])[:43]:45s} | {str(t['authors'])[:33]:35s}\n")
print("Done writing ranks_50_110.txt")
