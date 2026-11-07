import pandas as pd

test = pd.read_csv(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\data\test_with_abstracts.csv").set_index('id')
with open(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code\src\v40_pids_info.txt", "w", encoding="utf-8") as f:
    for pid in [1006, 1003, 1007, 1419, 751, 1372, 1053, 877, 1332, 885]:
        t = test.loc[pid]
        f.write(f"ID {pid:4d} | Venue: {t['venue']} | Year: {t['year']} | Title: {t['title']} | Authors: {str(t['authors'])}\n")
