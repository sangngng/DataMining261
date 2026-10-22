import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def extract_and_save_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    print(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_titles = train_df['title'].fillna("").tolist()
    test_titles = test_df['title'].fillna("").tolist()
    
    print(f"Found {len(train_titles)} train titles, {len(test_titles)} test titles.")
    print(f"Initializing pre-trained SentenceTransformer: '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    print("Encoding train titles...")
    train_embeddings = model.encode(train_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding test titles...")
    test_embeddings = model.encode(test_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    train_out = os.path.join(data_dir, "train_embeddings_minilm.npy")
    test_out = os.path.join(data_dir, "test_embeddings_minilm.npy")
    
    np.save(train_out, train_embeddings)
    np.save(test_out, test_embeddings)
    
    print(f"\n[SUCCESS]:")
    print(f" - Train embeddings saved: {train_out} with shape {train_embeddings.shape}")
    print(f" - Test embeddings saved:  {test_out} with shape {test_embeddings.shape}")

if __name__ == "__main__":
    extract_and_save_embeddings()
