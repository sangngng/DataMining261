import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("source-code/data")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts, tokenizer, model, device, batch_size=64):
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**encoded)
            embs = mean_pooling(out, encoded["attention_mask"])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)} texts...", flush=True)
    return np.vstack(all_embs)

def main():
    print("="*75, flush=True)
    print("=== EXTRACTING SEMANTIC K-MEANS TOPIC CLUSTERS (k=12) ===", flush=True)
    print("="*75, flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_df = pd.read_csv(DATA_DIR / "train_with_abstracts.csv")
    test_df  = pd.read_csv(DATA_DIR / "test_with_abstracts.csv")

    def make_doc(row):
        t = str(row["title"]) if pd.notna(row["title"]) else ""
        a = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
        v = str(row["venue"]) if pd.notna(row["venue"]) else ""
        return f"{v.upper()}: {t}. {a[:300]}"

    train_docs = [make_doc(r) for _, r in train_df.iterrows()]
    test_docs  = [make_doc(r) for _, r in test_df.iterrows()]

    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    print("Encoding train docs...", flush=True)
    tr_embs = encode_texts(train_docs, tokenizer, model, device)
    print("Encoding test docs...", flush=True)
    te_embs = encode_texts(test_docs, tokenizer, model, device)

    # Fit KMeans with 12 clusters
    k = 12
    print(f"Fitting KMeans with {k} clusters...", flush=True)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
    tr_clusters = kmeans.fit_predict(tr_embs)
    te_clusters = kmeans.predict(te_embs)

    tr_dist = kmeans.transform(tr_embs)
    te_dist = kmeans.transform(te_embs)

    # Analyze clusters
    vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1,2))
    tfidf_mat = vec.fit_transform(train_docs)
    vocab = np.array(vec.get_feature_names_out())

    print("\n" + "-"*75, flush=True)
    print(f"{'Cluster':<9} | {'Size':<6} | {'Mean Label':<10} | {'Top Keywords':<45}", flush=True)
    print("-"*75, flush=True)

    cluster_info = []
    y_tr = train_df["Label"].values
    for c in range(k):
        idx = np.where(tr_clusters == c)[0]
        c_size = len(idx)
        mean_lbl = y_tr[idx].mean() if c_size > 0 else 0.0
        
        c_tfidf_mean = np.asarray(tfidf_mat[idx].mean(axis=0)).flatten()
        top_kw_idx = np.argsort(c_tfidf_mean)[::-1][:5]
        top_kws = ", ".join(vocab[top_kw_idx])
        
        print(f"C{c:<8} | {c_size:<6} | {mean_lbl:<10.3f} | {top_kws:<45}", flush=True)
        cluster_info.append({"cluster": c, "size": c_size, "mean_label": mean_lbl, "keywords": top_kws})

    out_npz = DATA_DIR / "clusters_k12.npz"
    np.savez_compressed(
        out_npz,
        tr_embs=tr_embs,
        te_embs=te_embs,
        tr_clusters=tr_clusters,
        te_clusters=te_clusters,
        tr_dist=tr_dist,
        te_dist=te_dist
    )
    print(f"\n[SUCCESS] Saved clustering features to {out_npz}", flush=True)

if __name__ == "__main__":
    main()
