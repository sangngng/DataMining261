import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

# Look at venue breakdown by rank buckets
buckets = [
    (1, 25), (26, 50), (51, 100), (101, 150), (151, 200),
    (201, 250), (251, 300), (301, 350), (351, 374)
]

print("Rank Bucket | LPNMR | KR | CAV | LICS | Total")
print("-" * 45)
for start, end in buckets:
    sub = v37_act.loc[start-1:end-1]
    venues = [test.loc[pid, 'venue'] for pid in sub['id']]
    lpnmr = venues.count('lpnmr')
    kr = venues.count('kr')
    cav = venues.count('cav')
    lics = venues.count('lics')
    print(f"Ranks {start:3d}-{end:3d} | {lpnmr:5d} | {kr:3d} | {cav:3d} | {lics:4d} | {len(sub):5d}")
