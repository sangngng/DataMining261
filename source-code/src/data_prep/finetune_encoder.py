# Colab (GPU): !python colab_finetune.py MODEL TAG
# Fine-tunes a pretrained encoder on train labels (ordinal targets P(y>=k), k=2..5) with the SAME CV folds as the
# local pipeline (cv_pack.csv), then reports whether adding it to the current blend helps. Also fits on all train -> test.
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

MODEL, TAG = sys.argv[1], sys.argv[2]
EPOCHS, LR, BS, MAXLEN = 5, 3e-5, 32, 48
dev = "cuda"
tr = pd.read_csv("cv_pack.csv"); te = pd.read_csv("test_pack.csv")
y = tr.y.values
txt = lambda d: (d.venue.str.upper() + " " + d.year.astype(str) + ": " + d.title.fillna("")).tolist()
Xtr, Xte = txt(tr), txt(te)
tok = AutoTokenizer.from_pretrained(MODEL)
T = np.stack([(y >= k) for k in (2, 3, 4, 5)], 1).astype(np.float32)


class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(MODEL); s.drop = nn.Dropout(0.1); s.head = nn.Linear(s.enc.config.hidden_size, 4)
    def forward(s, **b):
        h = s.enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        return s.head(s.drop((h * m).sum(1) / m.sum(1)))


def enc(texts):
    return tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")


def fit_predict(tr_idx, pred_texts, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr_idx) / BS)); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(); lossf = nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), BS):
            bi = perm[i:i + BS]; b = {k: v.to(dev) for k, v in enc([Xtr[j] for j in bi]).items()}
            with torch.cuda.amp.autocast():
                loss = lossf(net(**b).float(), torch.tensor(T[bi]).to(dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(pred_texts), 128):
            b = {k: v.to(dev) for k, v in enc(pred_texts[i:i + 128]).items()}
            with torch.cuda.amp.autocast():
                out.append(torch.sigmoid(net(**b).float()).mean(1).cpu().numpy())
    del net; torch.cuda.empty_cache()
    return np.concatenate(out)


# ---- metric + venue-matched simulation (same definition as src/evaluation.py)
rk = rankdata(tr.id.astype(str).values, method="ordinal")
quota = {v: int(round(298 * c / len(te))) for v, c in te.venue.value_counts().items()}
pools = {v: np.where(tr.venue.values == v)[0] for v in quota}
def metric(yy, p, r):
    m = p > 0; o = np.lexsort((r[m], -p[m])); ys = yy[m][o]; w = 1 / np.log2(np.arange(1, len(ys) + 1) + 1)
    Q = (w * (ys - 1) / 4).sum() / w.sum(); R = (ys > 1).sum() / (yy > 1).sum(); return 2 * Q * R / (Q + R)
def prio(u, keep=0.65):
    n = len(u); m = int(round(keep * n)); o = np.argsort(-u, kind="stable"); p = np.zeros(n); p[o[:m]] = np.linspace(10, 0.01, m); return p
def simulate(cands, n=600, seed=5):
    rng = np.random.default_rng(seed); res = {k: [] for k in cands}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pools[v], q, replace=False) for v, q in quota.items()]); r = rng.integers(2)
        for k, U in cands.items(): res[k].append(metric(y[idx], prio(U[r][idx]), rk[idx]))
    return {k: np.array(v) for k, v in res.items()}

t0 = time.time(); oof = np.zeros((2, len(tr)))
for rep in (0, 1):
    f = tr[f"fold{rep}"].values
    for k in range(5):
        a, b = np.where(f != k)[0], np.where(f == k)[0]
        oof[rep, b] = fit_predict(a, [Xtr[j] for j in b], seed=100 * rep + k)
        print(f"rep {rep} fold {k} auc {roc_auc_score(y[b] > 1, oof[rep, b]):.4f}  {time.time() - t0:.0f}s", flush=True)
np.save(f"ft_{TAG}_oof.npy", oof.astype(np.float32))
rank01 = lambda v: rankdata(v) / len(v)
cur = np.stack([tr.cur0.values, tr.cur1.values]); ft = np.stack([rank01(oof[0]), rank01(oof[1])])
c = {"ft_alone": ft, "current": cur, "current+ft(1/4)": 0.75 * cur + 0.25 * ft, "current+ft(1/2)": 0.5 * cur + 0.5 * ft}
s = simulate(c)
print("RESULT", TAG, "AUC ft", [round(roc_auc_score(y > 1, oof[r]), 4) for r in (0, 1)], "AUC cur", [round(roc_auc_score(y > 1, cur[r]), 4) for r in (0, 1)],
      "AUC mix1/4", [round(roc_auc_score(y > 1, c["current+ft(1/4)"][r]), 4) for r in (0, 1)])
for k, v in s.items():
    d = v - s["current"]; print(f"RESULT {TAG} {k:18s} score {v.mean():.4f}  diff {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] P>0 {np.mean(d > 0):.2f}", flush=True)
print("RESULT spearman ft-cur", round(pd.Series(ft[0]).corr(pd.Series(cur[0]), method="spearman"), 3))
test = np.mean([fit_predict(np.arange(len(tr)), Xte, seed=1000 + s_) for s_ in range(3)], 0)
np.save(f"ft_{TAG}_test.npy", test.astype(np.float32)); print("RESULT test saved", TAG, f"{time.time() - t0:.0f}s")
