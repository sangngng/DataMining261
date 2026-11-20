"""Write a submission from the pipeline utilities (src/pipelines.py).
    python src/make_submission.py b      -> submissions/submission_b_keep60_358.csv        (pipeline B alone, keep 60%)
    python src/make_submission.py ab     -> submissions/submission_avg_ab_keep65_387.csv   (equal average of A and B)
    python src/make_submission.py abc    -> submissions/submission_avg_abc_keep65_387.csv  (equal average of A, B and C; final)
The top KEEP fraction gets distinct priorities 10 -> 0.01 in utility order; all other papers get 0 (not recommended)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from pipelines import test, r01
MODE = sys.argv[1]
te = pd.read_csv("work/test.csv"); ss = pd.read_csv("work/sample_submission.csv")
if MODE == "b":
    u, KEEP, name = test("B", ranked=False), 0.60, "b"
elif MODE == "ab":
    u, KEEP, name = 0.5 * test("A") + 0.5 * test("B"), 0.65, "avg_ab"
elif MODE == "abc":
    u, KEEP, name = (test("A") + test("B") + test("C")) / 3, 0.65, "avg_abc"
u = u + 1e-9 * np.arange(len(te))[::-1] / len(te)          # deterministic tie-break
m = int(round(KEEP * len(u))); o = np.argsort(-u, kind="stable"); p = np.zeros(len(u)); p[o[:m]] = np.linspace(10, 0.01, m)
sub = pd.DataFrame({"id": te.id.values, "priority": np.round(p, 6)}); act = sub.priority[sub.priority > 0]
assert len(sub) == 596 and set(sub.id) == set(ss.id) and sub.id.is_unique and np.isfinite(sub.priority).all() and (sub.priority >= 0).all() and act.is_unique
out = f"submissions/submission_{name}_keep{int(KEEP * 100)}_{m}.csv"; sub.to_csv(out, index=False); print("wrote", out, "recommended", m, "excluded", 596 - m)
