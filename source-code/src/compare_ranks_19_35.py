import pandas as pd

v37 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v37_pinnacle_apex_374.csv")
v39 = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\submissions\submission_v39_pure_consensus_apex_374.csv")
test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')

v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v39_act = v39[v39['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)

print(f"{'Rank':4s} | {'V37 ID':6s} | {'Venue':5s} | {'V37 Title':35s} | {'V39 ID':6s} | {'Venue':5s} | {'V39 Title':35s}")
print("-" * 115)
for r in range(19, 36):
    id37 = v37_act.loc[r-1, 'id']
    id39 = v39_act.loc[r-1, 'id']
    t37 = test.loc[id37]
    t39 = test.loc[id39]
    print(f"{r:4d} | {id37:6d} | {t37['venue']:5s} | {str(t37['title'])[:33]:35s} | {id39:6d} | {t39['venue']:5s} | {str(t39['title'])[:33]:35s}")
