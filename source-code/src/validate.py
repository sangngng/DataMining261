"""Local validation: repeated stratified-group CV with nested out-of-fold stacking, evaluated with the competition
metric on venue-matched random 298-paper splits.

    python -m src.validate cache --rep 0          # one CV repeat (seed 42+rep), ~90 s; reps 0-2 = development, 3-4 = fresh
    python -m src.validate report --reps 0 1 2    # tables, ablation, keep-fraction curves, label-shift robustness, errors
    python -m src.validate compare --reps 3 4     # 18-learner vs 23-learner stack on seeds not used for development
"""
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load, text_of, near_dup_groups, repeated_folds, ROOT, SEED
from .features import Space
from .models import BASE_NAMES, base_fit_predict, inner_oof, meta_matrix, stack_parts, blend, rank01
from .evaluation import VenueMatchedSampler

ART = ROOT / "artifacts"; REP = ROOT.parent / "reports"
GROUPS = {"text (TF-IDF/char/embeddings/kNN/GBDT)": ["lr_word", "ridge_word", "ridge_emb", "knn_g", "knn_p", "lr_char", "ridge_word_emb",
                                                      "gbdt_svd", "ridge_emb_vi", "lr_title_word", "ridge_char", "ord_lr", "mlp_emb", "lr_abs_char"],
          "authors": ["auth_mean", "auth_max", "auth_cnt"], "venue-year prior": ["venyr_prior"],
          "clustering target-encoding": ["clus12_te", "clus40_te"], "itemset / association rules": ["assoc_max", "assoc_min", "assoc_mean"]}


def cache(rep):
    tr, te, Etr, Ete = load(); S = Space(tr, te, Etr, Ete); y = tr["Label"].values
    groups = near_dup_groups(text_of(tr), 0.8); out = []
    for r, k, a, b in repeated_folds(y, groups, 5, rep + 1):
        if r != rep: continue
        out.append(dict(r=r, k=k, a=a, b=b, Btr=inner_oof(S, a, y[a], seed=SEED + r), Bva=base_fit_predict(S, a, y[a], b)))
    ART.mkdir(exist_ok=True); pickle.dump(out, open(ART / f"cv_rep{rep}.pkl", "wb")); print("saved rep", rep)


def load_reps(reps):
    folds = []
    for i, r in enumerate(reps):
        for f in pickle.load(open(ART / f"cv_rep{r}.pkl", "rb")): f["i"] = i; folds.append(f)
    return folds


def oof_parts(S, y, folds, cols, n_rep):
    P = {k: np.zeros((n_rep, len(y))) for k in ("ridge_reg", "ridge_clf", "lgb_reg", "lgb_clf")}
    for f in folds:
        p = stack_parts(meta_matrix(S, f["a"], f["Btr"], cols), y[f["a"]], meta_matrix(S, f["b"], f["Bva"], cols), len(cols), 1, SEED + f["r"])
        for k in P: P[k][f["i"], f["b"]] = p[k]
    return P


def stack_util(S, y, folds, cols, n_rep):
    P = oof_parts(S, y, folds, cols, n_rep)
    return np.stack([blend({k: v[i] for k, v in P.items()}) for i in range(n_rep)])


def setup(reps):
    tr, te, Etr, Ete = load(); S = Space(tr, te, Etr, Ete); y = tr["Label"].values
    return tr, te, S, y, load_reps(reps), VenueMatchedSampler(tr, te, y)


def compare(reps):
    tr, te, S, y, folds, sim = setup(reps); n = len(reps)
    u18 = stack_util(S, y, folds, list(range(18)), n); u23 = stack_util(S, y, folds, list(range(len(BASE_NAMES))), n)
    print(f"seeds {reps}: 18-learner vs 23-learner stack (paired, venue-matched 298 splits)")
    for f in (0.60, 0.65, 0.70):
        m18 = sim.curve(u18, [f])[f][0][0]; m23 = sim.curve(u23, [f])[f][0][0]; d = sim.paired(u23, u18, f)
        print(f"  keep {f:.2f}: 18 -> {m18:.4f} | 23 -> {m23:.4f} | diff {d[0]:+.4f} 95% [{d[1]:+.4f},{d[2]:+.4f}] P(>0)={d[3]:.2f}")


def report(reps):
    tr, te, S, y, folds, sim = setup(reps); n = len(reps); allc = list(range(len(BASE_NAMES))); L = []
    P = lambda *a: (print(*a), L.append(" ".join(str(x) for x in a)))
    u = stack_util(S, y, folds, allc, n)
    oof = np.zeros((n, len(y), len(BASE_NAMES)))
    for f in folds: oof[f["i"], f["b"]] = f["Bva"]
    rows = []
    cands = {"Random order": np.random.default_rng(0).random((n, len(y))), "Venue-year prior": oof[:, :, BASE_NAMES.index("venyr_prior")],
             "Author target-encoding": oof[:, :, BASE_NAMES.index("auth_mean")], "Clustering TE (k=40)": oof[:, :, BASE_NAMES.index("clus40_te")],
             "Itemset rules (mean lift)": oof[:, :, BASE_NAMES.index("assoc_mean")], "TF-IDF word + LogReg": oof[:, :, 0],
             "MiniLM embedding + Ridge": oof[:, :, 2], "MiniLM kNN": oof[:, :, 3], "Char TF-IDF + LogReg": oof[:, :, BASE_NAMES.index("lr_char")],
             "Ordinal LogReg P(y>=k)": oof[:, :, BASE_NAMES.index("ord_lr")], "Stack (ridge + LightGBM)": u}
    for name, uu in cands.items():
        for f in (0.65, 1.0):
            m, s = sim.curve(uu, [f], n_sims=500)[f]
            rows.append(dict(model=name, keep=f, score=m[0], Q=m[1], R=m[2], score_std=s[0]))
    df = pd.DataFrame(rows).round(4); REP.mkdir(exist_ok=True); df.to_csv(REP / "cv_results.csv", index=False)
    P("== model comparison (venue-matched 298 splits, same folds) ==\n" + df.to_string(index=False))
    P("\n== keep-fraction curve, stack ==")
    fr = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]; c = sim.curve(u, fr)
    for f in fr: P(f"  keep {f:.2f}: score {c[f][0][0]:.4f}  Q {c[f][0][1]:.4f}  R {c[f][0][2]:.4f}  (std over splits {c[f][1][0]:.4f})")
    P("\n== ablation (drop a group of base learners, refit stacker) ==")
    for g, names in GROUPS.items():
        cols = [i for i, nm in enumerate(BASE_NAMES) if nm not in names]
        ua = stack_util(S, y, folds, cols, n); P(f"  without {g:42s}: {sim.curve(ua, [0.65])[0.65][0][0]:.4f}")
    P("\n== label-shift robustness (odds multiplier on worthwhile papers in the simulated split) ==")
    for odds in (1.0, 1.4, 2.0, 3.0):
        cc = sim.curve(u, [0.6, 0.65, 0.7, 0.75, 0.8, 0.9], odds=odds); P(f"  odds x{odds}: " + "  ".join(f"{f}:{cc[f][0][0]:.4f}" for f in cc))
    P("\n== uncertainty: stack @65% vs best single learner (char TF-IDF LogReg), paired ==")
    d = sim.paired(u, cands["Char TF-IDF + LogReg"], 0.65); P(f"  diff {d[0]:+.4f}  95% [{d[1]:+.4f},{d[2]:+.4f}]  P(>0)={d[3]:.2f}")
    # error analysis (first repeat, 65% kept inside each venue pool)
    r0 = u[0]; keep = np.zeros(len(y), bool)
    for v, ix in sim.pools.items(): keep[ix[np.argsort(-r0[ix])[:int(.65 * len(ix))]]] = True
    df2 = tr.assign(u=r0, keep=keep, y=y)[tr["venue"].isin(sim.quota)]
    P("\n== error analysis ==")
    for v, g in df2.groupby("venue"):
        P(f"  {v:6s} n={len(g):4d} R={g[g.y > 1].keep.mean():.3f}  label-1 kept={g[g.y == 1].keep.mean():.3f}  mean label kept/dropped={g[g.keep].y.mean():.2f}/{g[~g.keep].y.mean():.2f}")
    P(f"  label-5 papers dropped: {int(((df2.y == 5) & ~df2.keep).sum())}/{int((df2.y == 5).sum())}; label-1 share in top 10%: "
      f"{(df2.sort_values('u', ascending=False).head(int(.1 * len(df2))).y == 1).mean():.3f} (base rate {(df2.y == 1).mean():.3f})")
    P("  false positives (label 1, ranked highest):")
    for _, r in df2[df2.y == 1].sort_values("u", ascending=False).head(6).iterrows(): P(f"    [{r.venue},{r.year}] {r.title[:90]}")
    P("  false negatives (label >= 4, ranked lowest):")
    for _, r in df2[df2.y >= 4].sort_values("u").head(6).iterrows(): P(f"    [{r.venue},{r.year},y={r.y}] {r.title[:90]}")
    (REP / "cv_summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cache", "report", "compare"])
    ap.add_argument("--rep", type=int, default=0); ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"cache": lambda: cache(a.rep), "report": lambda: report(a.reps), "compare": lambda: compare(a.reps)}[a.cmd]()
