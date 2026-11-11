"""Generate the final `id,priority` files. Fully algorithmic: no test id is referenced, no Public-LB feedback is used.

    python -m src.make_submission --keep 0.65 0.60 0.70            # full 23-learner stack
    python -m src.make_submission --bases 18 --keep 0.65            # earlier 18-learner stack
    python -m src.make_submission --model blend --keep 0.65         # final: stack + title-embedding learners, equal thirds
"""
import argparse
import numpy as np
import pandas as pd
from .data import load, DATA, ROOT, SEED
from .features import Space
from .models import BASE_NAMES, base_fit_predict, inner_oof, meta_matrix, stack_parts, blend
from .evaluation import priority_from_utility
from .embed_models import load_title_embeddings, title_emb_utility
from .models import rank01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, nargs="+", default=[0.65], help="fraction of the 596 test papers to recommend")
    ap.add_argument("--bases", type=int, default=len(BASE_NAMES), help="number of base learners (first N of BASE_NAMES)")
    ap.add_argument("--model", choices=["stack", "blend", "blend_ft"], default="stack",
                    help="stack = 23-learner stack only; blend = equal-weight rank average of stack, MiniLM-title and BGE-title learners; blend_ft = blend plus fine-tuned BGE and E5 (precomputed test predictions in data/ft_*_test.npy, see src/data_prep/finetune_encoder.py), five equal parts")
    ap.add_argument("--out", default=str(ROOT / "submissions"))
    args = ap.parse_args()

    tr, te, Etr, Ete = load(); S = Space(tr, te, Etr, Ete)
    y = tr["Label"].values; a = np.arange(len(tr)); b = len(tr) + np.arange(len(te))
    cols = list(range(args.bases))
    Btr = inner_oof(S, a, y, seed=SEED); Bte = base_fit_predict(S, a, y, b)       # inner-OOF inputs for the stacker
    parts = stack_parts(meta_matrix(S, a, Btr, cols), y, meta_matrix(S, b, Bte, cols), len(cols), n_seeds=5, seed=SEED)
    u = blend(parts)
    name = f"stack{args.bases}"
    if args.model == "blend":
        u = (rank01(u) + sum(title_emb_utility(S, load_title_embeddings(), a, y, b))) / 3; name = f"blend{args.bases}"
    if args.model == "blend_ft":
        ft = [rank01(np.load(DATA / f"ft_{t}_test.npy")) for t in ("bge", "e5")]
        u = (rank01(u) + sum(title_emb_utility(S, load_title_embeddings(), a, y, b)) + sum(ft)) / 5; name = f"blendft{args.bases}"
    u = u + 1e-9 * np.arange(len(te))[::-1] / len(te)                    # deterministic tie-break
    ss = pd.read_csv(DATA / "sample_submission.csv")
    for keep in args.keep:
        sub = pd.DataFrame({"id": te["id"].values, "priority": np.round(priority_from_utility(u, keep), 6)})
        act = sub.priority[sub.priority > 0]
        assert len(sub) == 596 and set(sub.id) == set(ss.id) and sub.id.is_unique
        assert np.isfinite(sub.priority).all() and (sub.priority >= 0).all() and act.is_unique
        path = f"{args.out}/submission_{name}_keep{int(round(keep * 100))}_{len(act)}.csv"
        sub.to_csv(path, index=False); print("wrote", path, "| recommended", len(act), "| excluded", 596 - len(act))


if __name__ == "__main__":
    main()
