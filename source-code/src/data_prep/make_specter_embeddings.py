"""Optional external embeddings (run once on a GPU machine, e.g. Google Colab; no labels are used).

    pip install sentence-transformers
    python -m src.data_prep.make_specter_embeddings --model BAAI/bge-base-en-v1.5
    python -m src.data_prep.make_specter_embeddings --model sentence-transformers/allenai-specter

Writes data/emb_<tag>_train.npy and data/emb_<tag>_test.npy (L2-normalised, rows in the same order as
train_with_abstracts.csv / test_with_abstracts.csv). Models must be cited in the report:
- SPECTER: Cohan et al., 2020, "SPECTER: Document-level Representation Learning using Citation-informed Transformers" (Apache-2.0)
- BGE: Xiao et al., 2023, "C-Pack: Packaged Resources To Advance General Chinese Embedding" (MIT)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def text_of(df):
    ab = df["abstract"].fillna("") if "abstract" in df else ""
    return (df["title"].fillna("") + ". " + ab.astype(str).str.slice(0, 600)).tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    tag = a.model.split("/")[-1].split("-")[0].lower()
    m = SentenceTransformer(a.model)
    for split in ("train", "test"):
        df = pd.read_csv(DATA / f"{split}_with_abstracts.csv")
        E = m.encode(text_of(df), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        np.save(DATA / f"emb_{tag}_{split}.npy", E.astype(np.float32)); print(split, E.shape)
