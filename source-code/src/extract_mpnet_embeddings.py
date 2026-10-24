import os
import sys
import html
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def clean_title(text):
    if pd.isna(text):
        return ""
    # Decode HTML entities (e.g. &apos;, &quot;, &amp;)
    cleaned = html.unescape(str(text))
    # Normalize extra whitespaces
    cleaned = " ".join(cleaned.split())
    return cleaned

def extract_and_save_mpnet():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")

    train_titles = [clean_title(t) for t in train_df['title']]
    test_titles = [clean_title(t) for t in test_df['title']]

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading all-mpnet-base-v2 on device: {device}...")
    t0 = time.time()
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    print(f"Loaded in {time.time()-t0:.2f}s. Embedding dim: {model.get_sentence_embedding_dimension()}")

    print("Encoding train titles (768d)...")
    train_emb = model.encode(train_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Train embeddings shape: {train_emb.shape}")

    print("Encoding test titles (768d)...")
    test_emb = model.encode(test_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Test embeddings shape: {test_emb.shape}")

    train_out = os.path.join(DATA_DIR, "train_embeddings_mpnet.npy")
    test_out = os.path.join(DATA_DIR, "test_embeddings_mpnet.npy")

    np.save(train_out, train_emb)
    np.save(test_out, test_emb)
    print(f"Successfully saved to:\n  {train_out}\n  {test_out}")

if __name__ == "__main__":
    extract_and_save_mpnet()
