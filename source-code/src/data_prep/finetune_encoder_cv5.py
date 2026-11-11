# Colab (GPU): !python colab_finetune.py MODEL TAG
# Fine-tunes a pretrained encoder on train labels (ordinal targets P(y>=k), k=2..5) with the SAME CV folds as the
# local pipeline (cv_pack5.csv), then reports whether adding it to the current blend helps. Also fits on all train -> test.
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

MODEL, TAG = sys.argv[1], sys.argv[2]
EPOCHS, LR, BS, MAXLEN = 5, 3e-5, 32, 48
dev = "cuda"
tr = pd.read_csv("cv_pack5.csv"); te = pd.read_csv("test_pack.csv")
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



REPS = (0, 1, 2, 3, 4)
t0 = time.time(); oof = np.zeros((len(REPS), len(tr)))
for ri, rep in enumerate(REPS):
    f = tr[f"fold{rep}"].values
    for k in range(5):
        a, b = np.where(f != k)[0], np.where(f == k)[0]
        oof[ri, b] = fit_predict(a, [Xtr[j] for j in b], seed=100 * rep + k)
        print(f"rep {rep} fold {k} auc {roc_auc_score(y[b] > 1, oof[ri, b]):.4f}  {time.time() - t0:.0f}s", flush=True)
    np.save(f"ft_{TAG}_oof5.npy", oof.astype(np.float32))
print("RESULT done", TAG, f"{time.time() - t0:.0f}s")
test = np.mean([fit_predict(np.arange(len(tr)), Xte, seed=1000 + s_) for s_ in range(3)], 0)
np.save(f"ft_{TAG}_test.npy", test.astype(np.float32)); print("RESULT test saved", TAG, f"{time.time() - t0:.0f}s")
