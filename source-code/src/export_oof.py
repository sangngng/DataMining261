"""Export the out-of-fold utility of the pipeline-A blend (stack + MiniLM-title + BGE-title learners, equal rank average)
for CV repeats 0-4, together with the fold assignment, so that other pipelines can be evaluated on identical folds.

    python -m src.validate cache --rep R      # for R in 0..4 (stack base learners)
    python -m src.export_oof                  # -> data/oof_blend_a.csv (id, y, venue, year, title, fold0..4, u0..4)
"""
import numpy as np, pandas as pd
from .validate import setup, stack_util
from .models import BASE_NAMES, rank01
from .embed_models import TAGS, load_title_embeddings, title_emb_predict
from .data import DATA

REPS = [0, 1, 2, 3, 4]


def main():
    tr, te, S, y, folds, sim = setup(REPS); n = len(REPS)
    U = stack_util(S, y, folds, list(range(len(BASE_NAMES))), n)
    embs = load_title_embeddings(); T = np.zeros((n, len(TAGS), 2, len(y))); F = np.full((n, len(y)), -1)
    for f in folds:
        for j, t in enumerate(TAGS):
            T[f["i"], j, 0, f["b"]], T[f["i"], j, 1, f["b"]] = title_emb_predict(S, *embs[t], f["a"], y[f["a"]], f["b"])
        F[f["i"], f["b"]] = f["k"]
    V = [[(rank01(T[i, j, 0]) + rank01(T[i, j, 1])) / 2 for j in range(len(TAGS))] for i in range(n)]
    out = tr[["id", "venue", "year", "title"]].assign(y=y)
    for i in range(n):
        out[f"fold{i}"] = F[i]; out[f"u{i}"] = (rank01(U[i]) + sum(V[i])) / (1 + len(TAGS))
    out.to_csv(DATA / "oof_blend_a.csv", index=False); print("wrote", DATA / "oof_blend_a.csv")


if __name__ == "__main__":
    main()
