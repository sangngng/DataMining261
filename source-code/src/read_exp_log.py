import pandas as pd

df = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\reports\experiment_log.csv")
for _, r in df.tail(15).iterrows():
    print(f"#{r['experiment_id']} | {r['model_name']} | LB: {r['public_lb_score']} | Notes: {str(r['notes'])[:80]}")
