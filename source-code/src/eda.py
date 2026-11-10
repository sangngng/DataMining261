"""Exploratory data analysis: figures -> reports/figures/eda_*.png, numbers -> reports/eda_summary.txt.
    python -m src.eda
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import load, ROOT, text_of, near_dup_groups

OUT = ROOT.parent / "reports"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def main():
    tr, te, Etr, Ete = load(); y = tr.Label.values; L = []
    P = lambda *a: (print(*a), L.append(" ".join(str(x) for x in a)))
    P("== sizes ==", len(tr), "train /", len(te), "test; total", len(tr) + len(te))
    P("== label distribution ==", tr.Label.value_counts().sort_index().to_dict(), "| worthwhile (y>1) share", round((y > 1).mean(), 3),
      "| mean gain", round(((y - 1) / 4).mean(), 3))
    P("== missing ==", {c: int(tr[c].isna().sum()) for c in ["title", "venue", "year", "authors", "doi"]},
      "| test authors missing", int(te.authors.isna().sum()), "| abstract coverage train/test", round(tr.abstract.notna().mean(), 3), round(te.abstract.notna().mean(), 3))
    g = near_dup_groups(text_of(tr), 0.8); sizes = np.bincount(g)
    P("== duplicates == duplicate ids", int(tr.id.duplicated().sum()), "| duplicate DOI", int(tr.doi.duplicated().sum()),
      "| near-duplicate title groups (cos>=0.8) with >1 paper:", int((sizes > 1).sum()), "covering", int(sizes[sizes > 1].sum()), "papers")
    P("== venue (train label rates) ==")
    t = tr.assign(w=y > 1).groupby("venue").agg(n=("w", "size"), worthwhile=("w", "mean"), mean_label=("Label", "mean")).round(3); P(t.to_string())
    P("== venue share train vs test ==", {v: (round((tr.venue == v).mean(), 3), round((te.venue == v).mean(), 3)) for v in sorted(set(tr.venue) | set(te.venue))})
    P("== year range train/test ==", (int(tr.year.min()), int(tr.year.max())), (int(te.year.min()), int(te.year.max())))
    A = lambda df: [set(a.split(", ")) - {""} if isinstance(a, str) else set() for a in df.authors]
    atr, ate = A(tr), A(te); known = set().union(*atr)
    P("== authors == unique in train", len(known), "| test papers with >=1 author seen in train", sum(bool(a & known) for a in ate), "/", len(te))
    cnt = Counter(a for s in atr for a in s); hi = {a: np.mean([y[i] for i, s in enumerate(atr) if a in s]) for a, c in cnt.items() if c >= 8}
    P("   authors (>=8 papers) with highest mean label:", sorted(hi.items(), key=lambda kv: -kv[1])[:5])
    P("   authors (>=8 papers) with lowest mean label:", sorted(hi.items(), key=lambda kv: kv[1])[:5])
    tl = tr.title.str.split().str.len(); P("== title length (words) corr with label ==", round(np.corrcoef(tl, y)[0, 1], 3), "| mean", round(tl.mean(), 1))
    ids = tr.sort_values("id"); gg = (ids.Label.values - 1) / 4
    P("== id order == lag-1 autocorrelation of labels along id:", round(np.corrcoef(gg[:-1], gg[1:])[0, 1], 3), "(ids follow venue/proceedings order)")
    # top discriminative words: high (y>=4) vs low (y==1)
    vec = TfidfVectorizer(stop_words="english", min_df=15, ngram_range=(1, 2)); X = vec.fit_transform(text_of(tr)); vocab = np.array(vec.get_feature_names_out())
    d = np.asarray(X[y >= 4].mean(0)).ravel() - np.asarray(X[y == 1].mean(0)).ravel()
    P("== words most associated with y>=4:", ", ".join(vocab[np.argsort(-d)[:15]])); P("== words most associated with y==1:", ", ".join(vocab[np.argsort(d)[:15]]))
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tr.Label.value_counts().sort_index().plot.bar(ax=ax[0], color="#4C72B0"); ax[0].set_title("Label distribution (train)")
    pd.crosstab(tr.venue, tr.Label, normalize="index").plot.bar(stacked=True, ax=ax[1], colormap="viridis"); ax[1].set_title("Label mix by venue")
    pd.crosstab(tr.year.clip(upper=2025), tr.Label, normalize="index").plot.bar(stacked=True, ax=ax[2], colormap="viridis"); ax[2].set_title("Label mix by year"); plt.tight_layout()
    plt.savefig(FIG / "eda_labels.png", dpi=130); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); vs = sorted(set(tr.venue) | set(te.venue))
    ax[0].bar(np.arange(len(vs)) - .2, [(tr.venue == v).mean() for v in vs], .4, label="train"); ax[0].bar(np.arange(len(vs)) + .2, [(te.venue == v).mean() for v in vs], .4, label="test")
    ax[0].set_xticks(range(len(vs))); ax[0].set_xticklabels(vs); ax[0].set_title("Venue share: train vs test (test has no iclp)"); ax[0].legend()
    yrs = list(range(2016, 2026)); ax[1].bar(np.array(yrs) - .2, [(tr.year == v).mean() for v in yrs], .4, label="train"); ax[1].bar(np.array(yrs) + .2, [(te.year == v).mean() for v in yrs], .4, label="test")
    ax[1].set_title("Year share: train vs test"); ax[1].legend(); plt.tight_layout(); plt.savefig(FIG / "eda_train_vs_test.png", dpi=130); plt.close()
    (OUT / "eda_summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
