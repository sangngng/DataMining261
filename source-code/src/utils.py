import os
import random
import numpy as np
import pandas as pd
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SUBMISSION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submissions")

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and sample submission files.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(data_dir, "sample_submission.csv")
    if not os.path.exists(sub_path):
        alt_sub = os.path.join(data_dir, "submission.csv")
        if os.path.exists(alt_sub):
            sub_path = alt_sub

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. "
            f"Please download train.csv and test.csv from Kaggle into {data_dir}."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else None

    return train_df, test_df, sample_sub

def save_submission(
    ids: list,
    priorities: list,
    filename: str,
    output_dir: str = SUBMISSION_DIR
) -> str:
    """
    Format and validate submission CSV:
      - 2 columns: id,priority
      - Exactly 596 rows
      - Finite non-negative numbers
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    sub = pd.DataFrame({
        'id': ids,
        'priority': priorities
    })

    # Validate
    assert len(sub) == 596, f"Expected 596 rows, got {len(sub)}"
    assert not sub['priority'].isna().any(), "Found NaN values in priority"
    assert (sub['priority'] >= 0).all(), "Priority scores must be non-negative"
    assert np.isfinite(sub['priority']).all(), "Priority scores must be finite"

    sub.to_csv(out_path, index=False)
    print(f" Saved valid submission file to: {out_path} ({len(sub)} rows)")
    return out_path
