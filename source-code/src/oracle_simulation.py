import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from metric import compute_competition_metric

def run_oracle_simulation(n_splits=5, seed=42):
    print("=" * 70)
    print(f"🔬 REPRODUCIBLE ORACLE BOUND SIMULATION ({n_splits}-Fold Stratified CV, seed={seed})")
    print("=" * 70)
    
    # 1. Load Data
    train_path = os.path.join(BASE_DIR, "data", "train.csv")
    emb_path = os.path.join(BASE_DIR, "data", "train_embeddings_minilm.npy")
    
    train_df = pd.read_csv(train_path)
    y = train_df['Label'].values
    ids = train_df['id'].values
    is_pos = (y >= 2).astype(int)
    
    emb = np.load(emb_path)
    N = len(train_df)
    
    # Selection ratio matching test set (381 / 596 = 63.926%)
    n_rec = int(np.round(N * 381.0 / 596.0)) # 1594 items
    print(f"Total training papers: {N}")
    print(f"Shortlist budget:     {n_rec} papers ({n_rec/N*100:.2f}%)\n")
    
    # 2. Compute 5-Fold OOF Predictions using Standard SBERT + LightGBM
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(N)
    oof_reg = np.zeros(N)
    
    for fold, (tr, val) in enumerate(skf.split(train_df, y)):
        # Stage 1 Classifier
        clf = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=16,
            random_state=seed + fold,
            verbose=-1
        )
        clf.fit(emb[tr], is_pos[tr])
        oof_prob[val] = clf.predict_proba(emb[val])[:, 1]
        
        # Stage 2 Regressor
        reg = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=16,
            random_state=seed + fold,
            verbose=-1
        )
        reg.fit(emb[tr], y[tr])
        oof_reg[val] = reg.predict(emb[val])
        
    # Model Selection: Pick top n_rec items by predicted probability
    model_shortlist_mask = np.zeros(N, dtype=bool)
    model_top_indices = np.argsort(oof_prob)[::-1][:n_rec]
    model_shortlist_mask[model_top_indices] = True
    
    # Oracle Selection: Pick top n_rec items by true label y (ties broken by oof_prob)
    oracle_shortlist_mask = np.zeros(N, dtype=bool)
    oracle_score_for_selection = y.astype(float) + 0.0001 * oof_prob
    oracle_top_indices = np.argsort(oracle_score_for_selection)[::-1][:n_rec]
    oracle_shortlist_mask[oracle_top_indices] = True
    
    # -------------------------------------------------------------------------
    # SCENARIO A: Realistic Model Selection + Model Ranking (Baseline)
    # -------------------------------------------------------------------------
    priorities_A = np.where(model_shortlist_mask, oof_reg, 0.0)
    score_A, q_A, r_A = compute_competition_metric(ids, y, priorities_A)
    
    # -------------------------------------------------------------------------
    # SCENARIO B: Realistic Model Selection + PERFECT Oracle Ranking
    # (Shows the maximum headroom achievable by ONLY re-ranking within the shortlist)
    # -------------------------------------------------------------------------
    priorities_B = np.where(model_shortlist_mask, y.astype(float) + 0.001 * oof_reg, 0.0)
    score_B, q_B, r_B = compute_competition_metric(ids, y, priorities_B)
    
    # -------------------------------------------------------------------------
    # SCENARIO C: PERFECT Oracle Selection (Zero False Positives) + Model Ranking
    # (Shows the headroom achievable by improving Stage 1 Classifier alone)
    # -------------------------------------------------------------------------
    priorities_C = np.where(oracle_shortlist_mask, oof_reg, 0.0)
    score_C, q_C, r_C = compute_competition_metric(ids, y, priorities_C)
    
    # -------------------------------------------------------------------------
    # SCENARIO D: Full Oracle (Perfect Selection AND Perfect Ranking)
    # (Theoretical mathematical ceiling of the metric)
    # -------------------------------------------------------------------------
    priorities_D = np.where(oracle_shortlist_mask, y.astype(float), 0.0)
    score_D, q_D, r_D = compute_competition_metric(ids, y, priorities_D)
    
    # 3. Print Comprehensive Benchmark Table
    print("-" * 75)
    print(f"{'Scenario':<42} | {'Score':<8} | {'Relevance Q':<11} | {'Coverage R':<10}")
    print("-" * 75)
    print(f"{'A. Model Selection + Model Ranking (Baseline)':<42} | {score_A:<8.5f} | {q_A:<11.5f} | {r_A:<10.5f}")
    print(f"{'B. Model Selection + ORACLE Ranking':<42} | {score_B:<8.5f} | {q_B:<11.5f} | {r_B:<10.5f}")
    print(f"{'C. ORACLE Selection + Model Ranking':<42} | {score_C:<8.5f} | {q_C:<11.5f} | {r_C:<10.5f}")
    print(f"{'D. Full Oracle (Selection + Ranking)':<42} | {score_D:<8.5f} | {q_D:<11.5f} | {r_D:<10.5f}")
    print("-" * 75)
    
    print("\n📈 HEADROOM ANALYSIS:")
    ranking_headroom = score_B - score_A
    selection_headroom = score_C - score_A
    total_headroom = score_D - score_A
    print(f"  * Headroom from Re-ranking alone (Scenario B - A):   +{ranking_headroom:.5f} ({ranking_headroom/total_headroom*100:.1f}% of total)")
    print(f"  * Headroom from Classification alone (Scenario C - A): +{selection_headroom:.5f} ({selection_headroom/total_headroom*100:.1f}% of total)")
    print(f"  * Ratio of Selection vs Ranking Impact:              {selection_headroom / max(ranking_headroom, 1e-6):.1f}x larger!")
    print("=" * 75)

if __name__ == "__main__":
    run_oracle_simulation()
