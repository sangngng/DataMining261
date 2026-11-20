"""Local CV comparison of the pipelines and their equal-weight averages on identical simulated 298-paper splits
(venue x year mix of the test set; 2500 splits; OOF repeats 0-1 of each pipeline's own folds).
    python src/compare.py   -> reports/cv_compare.txt
diff / P>0 are paired against the reference row (the A+B average)."""
import sys
sys.path.insert(0, "src")
from evaluate import evaluate
from pipelines import oof
A, B, C, D = (oof(k) for k in "ABCD")
cands = {"A (source-code, blend_ft4)": A, "B": B, "C (B code, other folds/seeds)": C, "D (C + ranking-loss encoders)": D,
         "avg A+B": (A + B) / 2, "avg A+B+C (final)": (A + B + C) / 3, "avg A+B+D": (A + B + D) / 3,
         "avg A+B+C+D": (A + B + C + D) / 4, "avg B+C": (B + C) / 2}
out = []
for k in (0.60, 0.65, 0.70):
    t = evaluate(cands, keep=k, n_sims=2500, seed=11, ref="avg A+B").to_string(index=False)
    print("keep", k); print(t); out += [f"keep {k}", t, ""]
open("reports/cv_compare.txt", "w").write("\n".join(out))
