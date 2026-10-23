import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

def extract_and_save_bge_embeddings():
    print("--- Extracting BAAI/bge-small-en-v1.5 Dense Embeddings ---")
    data_dir = os.path.join(BASE_DIR, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_titles = train_df['title'].fillna('').tolist()
    test_titles = test_df['title'].fillna('').tolist()

    print(f"Loading BAAI/bge-small-en-v1.5 model...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    print(f"Encoding {len(train_titles)} train titles...")
    t0 = time.time()
    train_emb = model.encode(train_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Train encoding done in {time.time() - t0:.2f}s. Shape: {train_emb.shape}")

    print(f"Encoding {len(test_titles)} test titles...")
    t0 = time.time()
    test_emb = model.encode(test_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Test encoding done in {time.time() - t0:.2f}s. Shape: {test_emb.shape}")

    train_out = os.path.join(data_dir, "train_embeddings_bge.npy")
    test_out = os.path.join(data_dir, "test_embeddings_bge.npy")

    np.save(train_out, train_emb)
    np.save(test_out, test_emb)
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")
    print("All BGE embeddings extracted and cached successfully!")

if __name__ == "__main__":
    extract_and_save_bge_embeddings()
