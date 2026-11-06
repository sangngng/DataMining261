"""
create_v39_pure_consensus.py:
Ultra-safe, surgically-anchored consensus submission:

Diagnostic & Design Principles:
1. RESTORE 100% FROZEN RANKS 1 TO 18 FROM V37 (0.66852 PB):
   - Preserves the proven NNV tool core (1920, 1648, 1835, 1921) at Ranks 11-14.
   - Preserves 1305 (PLP) & 1296 (ASP) at Ranks 15-16.
   - Preserves 1877 (Guy Katz NNV) & 1393 (Fichte ASP) at Ranks 17-18.
2. RESTORE 100% FROZEN RANKS 36 TO 374 FROM V37:
   - Absolutely NO teleportation of mid-tier benchmark papers (1275, 1318) into the top 25.
   - All tail noise filters (medical, hardware ESBMC, timed automata) preserved at Ranks 70-80.
3. MICRO-OPTIMIZE RANKS 19 TO 35 ONLY:
   - Permutes strictly within the 17 papers of Ranks 19-35 based on verified training gain:
     - Rank 19: 1700 (Guy Katz & Komendantskaya, CAV 2022, train mean 4.75)
     - Rank 20: 1925 (Matthew B. Dwyer & David Shriver, CAV 2020, 100% Label 5)
     - Rank 21: 1626 (Vehicle Tutorial, CAV 2023, train mean 4.50)
     - Rank 22:  878 (Zeynep Saribatur, KR 2021, train mean 4.50)
     - Rank 23: 1405 (Christoph Redl, LPNMR 2017, 100% Label 5)
     - Rank 24:  903 (Kewen Wang, KR 2020, train mean 4.33)
     - Rank 25:  989 (Artur d'Avila Garcez, KR 2020, train mean 4.00)
     - Rank 26: 1123 (Riccardo Rosati & Domenico Lembo, KR 2018, train mean 4.00)
     - Rank 27: 1207 (David Poole, KR 2016, train mean 4.00)
     - Rank 28: 1337 (Peñaloza & Ricca, LPNMR 2022)
     - Rank 29: 1321 (Thomas Eiter, LPNMR 2022)
     - Rank 30: 1502 (Maire & Genet, CAV 2025)
     - Rank 31: 1710 (Imandra Neural Networks, CAV 2022)
     - Rank 32: 1329 (Yuliya Lierler, LPNMR 2022)
     - Rank 33: 1302 (Mario Alviano, LPNMR 2022, train mean 2.73)
     - Rank 34:  911 (Mario Alviano, KR 2020, train mean 2.73)
     - Rank 35: 1122 (Gerhard Brewka, KR 2018, train mean 2.67)
4. Strict zero-tie monotonic convex priority mapping over the exact 374 budget.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUB_DIR = BASE_DIR / "submissions"

def save_submission(test_df, active_order, M, out_path):
    final_prio = np.zeros(len(test_df), dtype=np.float64)
    for rank_0, p_id in enumerate(active_order):
        g_idx = test_df[test_df["id"] == p_id].index[0]
        convex_curve = ((M - 1 - rank_0) / max(M - 1, 1)) ** 1.35
        linear_step = (M - rank_0) * 1e-4
        final_prio[g_idx] = 0.050000 + 1.200000 * convex_curve + linear_step

    final_prio = np.round(final_prio, 6)
    sub_df = pd.DataFrame({'id': test_df['id'], 'priority': final_prio})

    # Strict Audits
    assert len(sub_df) == 596
    assert (sub_df['id'] == test_df['id']).all()
    assert (sub_df['priority'] > 0).sum() == M
    assert (sub_df['priority'] == 0).sum() == 596 - M
    assert len(sub_df['priority'].unique()) == M + 1, "Duplicate priority values found!"

    sub_df.to_csv(out_path, index=False)
    print(f"SUCCESS: Generated {out_path} (Active={M}, Pruned={596-M}, Zero Ties Guaranteed!)")

def generate_v39():
    test = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")
    v37 = pd.read_csv(SUB_DIR / "submission_v37_pinnacle_apex_374.csv").set_index('id').reindex(test['id']).reset_index()

    v37_active = v37[v37['priority'] > 0].sort_values('priority', ascending=False).reset_index(drop=True)
    order_374 = list(v37_active['id'].values)
    assert len(order_374) == 374

    # Ranks 1 to 18: 100% Frozen from V37
    frozen_top18 = order_374[:18]

    # Ranks 36 to 374: 100% Frozen from V37
    frozen_tail = order_374[35:]

    # Ranks 19 to 35: Pure quality re-ordering
    new_middle = [
        1700, # Rank 19: Guy Katz & Komendantskaya (CAV 2022)
        1925, # Rank 20: Dwyer & Shriver (CAV 2020)
        1626, # Rank 21: Vehicle Tutorial (CAV 2023)
         878, # Rank 22: Zeynep Saribatur (KR 2021)
        1405, # Rank 23: Christoph Redl (LPNMR 2017)
         903, # Rank 24: Kewen Wang (KR 2020)
         989, # Rank 25: Artur Garcez (KR 2020)
        1123, # Rank 26: Riccardo Rosati (KR 2018)
        1207, # Rank 27: David Poole (KR 2016)
        1337, # Rank 28: Peñaloza & Ricca (LPNMR 2022)
        1321, # Rank 29: Thomas Eiter (LPNMR 2022)
        1502, # Rank 30: Maire & Genet CHCs (CAV 2025)
        1710, # Rank 31: Imandra Neural Networks (CAV 2022)
        1329, # Rank 32: Yuliya Lierler (LPNMR 2022)
        1302, # Rank 33: Mario Alviano (LPNMR 2022)
         911, # Rank 34: Mario Alviano (KR 2020)
        1122  # Rank 35: Gerhard Brewka (KR 2018)
    ]

    assert set(new_middle) == set(order_374[18:35]), "Middle set mismatch!"

    final_order = frozen_top18 + new_middle + frozen_tail
    assert len(final_order) == 374
    assert set(final_order) == set(order_374)

    out_file = SUB_DIR / "submission_v39_pure_consensus_apex_374.csv"
    save_submission(test, final_order, 374, out_file)

if __name__ == '__main__':
    generate_v39()
