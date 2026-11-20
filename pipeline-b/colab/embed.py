# Google Colab (T4 GPU). Frozen title embeddings for src/models_emb.py.
#   Upload work/train.csv and work/test.csv, then:  !python embed.py   -> emb_{minilm,bge}_{train,test}.npy (copy to work/)
# Encoders: sentence-transformers/all-MiniLM-L6-v2 (Reimers & Gurevych, 2019; Apache-2.0),
#           BAAI/bge-base-en-v1.5 (Xiao et al., 2023; MIT). Title only, L2-normalised. No labels are used.
import time, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); t0 = time.time()
for tag, name in (("minilm", "sentence-transformers/all-MiniLM-L6-v2"), ("bge", "BAAI/bge-base-en-v1.5")):
    m = SentenceTransformer(name, device="cuda")
    for part, d in (("train", tr), ("test", te)):
        np.save(f"emb_{tag}_{part}.npy", m.encode(d.title.tolist(), batch_size=64, normalize_embeddings=True).astype(np.float32))
    print("emb", tag, f"{time.time()-t0:.0f}s", flush=True)
