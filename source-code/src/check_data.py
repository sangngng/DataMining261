import os
import shutil
import zipfile
import glob
import pandas as pd

DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
PROJECT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def find_and_setup_dataset():
    os.makedirs(PROJECT_DATA_DIR, exist_ok=True)
    
    # 1. Check if files already in data/
    train_dest = os.path.join(PROJECT_DATA_DIR, "train.csv")
    test_dest = os.path.join(PROJECT_DATA_DIR, "test.csv")
    
    if os.path.exists(train_dest) and os.path.exists(test_dest):
        print(f" Dataset already found in: {PROJECT_DATA_DIR}")
        _verify_dataset(train_dest, test_dest)
        return True

    # 2. Check for zip files in Downloads
    zip_candidates = glob.glob(os.path.join(DOWNLOADS_DIR, "*mining*.zip")) + \
                     glob.glob(os.path.join(DOWNLOADS_DIR, "*assignment*.zip")) + \
                     glob.glob(os.path.join(DOWNLOADS_DIR, "*.zip"))
    
    found_zip = None
    for z in zip_candidates:
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                names = zip_ref.namelist()
                if any("train.csv" in n.lower() for n in names) and any("test.csv" in n.lower() for n in names):
                    found_zip = z
                    break
        except Exception:
            continue

    if found_zip:
        print(f" Found dataset zip in Downloads: {found_zip}")
        print(" Extracting into data/ ...")
        with zipfile.ZipFile(found_zip, 'r') as zip_ref:
            zip_ref.extractall(PROJECT_DATA_DIR)
        print(" Extraction complete!")
        _verify_dataset(train_dest, test_dest)
        return True

    # 3. Check for raw CSV files in Downloads
    train_dl = os.path.join(DOWNLOADS_DIR, "train.csv")
    test_dl = os.path.join(DOWNLOADS_DIR, "test.csv")
    sub_dl = os.path.join(DOWNLOADS_DIR, "sample_submission.csv")
    alt_sub_dl = os.path.join(DOWNLOADS_DIR, "submission.csv")

    if os.path.exists(train_dl) and os.path.exists(test_dl):
        print(" Found unzipped train.csv and test.csv in Downloads. Copying to data/ ...")
        shutil.copy(train_dl, train_dest)
        shutil.copy(test_dl, test_dest)
        if os.path.exists(sub_dl):
            shutil.copy(sub_dl, os.path.join(PROJECT_DATA_DIR, "sample_submission.csv"))
        elif os.path.exists(alt_sub_dl):
            shutil.copy(alt_sub_dl, os.path.join(PROJECT_DATA_DIR, "sample_submission.csv"))
        print(" Files copied!")
        _verify_dataset(train_dest, test_dest)
        return True

    print("\n[INFO] Dataset files not found yet.")
    print(" Please click 'Download All' on the Kaggle Data tab in Safari.")
    print(" Once downloaded, run: python -m src.check_data to automatically setup.")
    return False

def _verify_dataset(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("\n--- Dataset Verification ---")
    print(f" Train shape: {train_df.shape} (Expected ~2494 rows)")
    print(f" Test shape:  {test_df.shape} (Expected 596 rows)")
    print(f" Train columns: {list(train_df.columns)}")
    print(f" Test columns:  {list(test_df.columns)}")
    if 'Label' in train_df.columns:
        print("\n Label distribution in train.csv:")
        print(train_df['Label'].value_counts().sort_index())

if __name__ == "__main__":
    find_and_setup_dataset()
