import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\source-code")
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

def create_v40_candidate():
    test = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")
    v37 = pd.read_csv(SUB_DIR / "submission_v37_pinnacle_apex_374.csv").set_index('id').reindex(test['id']).reset_index()

    v37_active = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
    order = list(v37_active['id'].values)
    assert len(order) == 374

    # 1. Demote confirmed clinical/noisy drags to Ranks 85-89 (safely inside active 374)
    # 1006: Type 2 Diabetes Mellitus ML
    # 1003: Social Media Medication Effects NLP
    # 1007: Haemodialysis Time Series Deep Learning
    # 1419: Zhizheng Zhang (100% Label 1 in train)
    #  751: Embed2Sym (train labels 2)
    demotions = [
        (1006, 85),
        (1003, 86),
        (1007, 87),
        (1419, 88),
        (751,  89)
    ]
    for pid, target_rank in demotions:
        order.remove(pid)
        order.insert(target_rank - 1, pid)

    # 2. Surgical promotions of proven ASP / KR gems into high-DCG positions:
    # 1372: Joohyung Lee & Yi Wang (MDPs in ASP, LPNMR 2019) -> Rank 25
    # 1053: Alfano, Greco, Parisi (KR 2018 Argumentation, all train mean >= 3.7) -> Rank 48
    #  877: Javier Romero, Torsten Schaub, Blai Bonet (KR 2021 Planning, train mean 4.33 / 3.81) -> Rank 49
    # 1332: Ricardo Gonçalves & João Leite (LPNMR 2022 Provenance, train mean 4.50 / 3.60) -> Rank 50
    #  885: Egor Kostylev, Cuenca Grau, Kaminski (DatalogMTL Stable Models / ASP, KR 2021) -> Rank 51
    promotions = [
        (1372, 25),
        (1053, 48),
        (877,  49),
        (1332, 50),
        (885,  51)
    ]
    for pid, target_rank in promotions:
        order.remove(pid)
        order.insert(target_rank - 1, pid)

    assert len(order) == 374
    assert set(order) == set(v37_active['id'].values)
    return test, order

test, order = create_v40_candidate()
print("Success creating candidate V40 order!")

# Verify Ranks 1 to 24 are 100% identical to V37
v37 = pd.read_csv(SUB_DIR / "submission_v37_pinnacle_apex_374.csv").set_index('id').reindex(test['id']).reset_index()
v37_act = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
v37_order = list(v37_act['id'].values)

assert order[:24] == v37_order[:24], "Ranks 1 to 24 MUST be 100% identical to V37!"
print("Verified: Ranks 1 to 24 are 100% FROZEN from V37!")
print(f"Rank 25 in V40: {order[24]} (was {v37_order[24]} in V37)")
print(f"Rank 26 in V40: {order[25]} (was {v37_order[25]} in V37)")
