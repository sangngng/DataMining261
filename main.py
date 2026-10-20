import os
import argparse
import numpy as np
import pandas as pd

from src.utils import set_seed, load_data, save_submission
from src.features import FeaturePipeline
from src.models import TwoStageRecommender
from src.evaluate import run_cross_validation

def main():
    parser = argparse.ArgumentParser(description="CO3029 Data Mining Assignment Pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sub-name", type=str, default="submission_two_stage.csv", help="Submission output filename")
    parser.add_argument("--clusters", type=int, default=8, help="Number of KMeans clusters")
    parser.add_argument("--tfidf-max", type=int, default=2500, help="Max TF-IDF features")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # 1. Load Data
    print("\n--- Step 1: Loading Dataset ---")
    try:
        train_df, test_df, sample_sub = load_data()
        print(f"Loaded train.csv: {train_df.shape} rows")
        print(f"Loaded test.csv:  {test_df.shape} rows")
    except FileNotFoundError as e:
        print(f"\n[ERROR]: {e}")
        print("Please place train.csv and test.csv inside the 'data/' directory to run.")
        return

    # 2. Run Cross Validation
    print("\n--- Step 2: Running Stratified 5-Fold Cross Validation ---")
    best_score, best_q, best_r, best_t = run_cross_validation(train_df, n_splits=5, seed=args.seed)

    # 3. Train on Full Training Set and Predict Test Set
    print("\n--- Step 3: Training on Full Data and Predicting Test Set ---")
    fe_full = FeaturePipeline(
        tfidf_max_features=args.tfidf_max,
        n_clusters=args.clusters,
        random_state=args.seed
    )
    X_train_full = fe_full.fit_transform(train_df)
    X_test = fe_full.transform(test_df)
    y_train_full = train_df['Label'].values

    model_full = TwoStageRecommender(random_state=args.seed, threshold=best_t)
    model_full.fit(X_train_full, y_train_full)

    # Predict priorities for test set
    test_priorities = model_full.predict_priority(X_test, threshold=best_t)
    
    # 4. Save Submission
    print("\n--- Step 4: Generating and Validating Submission File ---")
    out_file = save_submission(
        ids=test_df['id'].tolist(),
        priorities=test_priorities.tolist(),
        filename=args.sub_name
    )

    # Quick inspection
    non_zero_count = np.sum(test_priorities > 0)
    print(f"Recommended papers (priority > 0): {non_zero_count} / {len(test_priorities)}")
    print(f"Filtered out papers (priority == 0): {len(test_priorities) - non_zero_count} / {len(test_priorities)}")
    print(f"\nPipeline completed successfully! Ready to submit {out_file} on Kaggle.")

if __name__ == "__main__":
    main()
