# Google Colab (T4 GPU). Fine-tuned title encoders for pipeline B/C.
#   Upload work/train.csv, work/test.csv and work/folds_b.csv / work/folds_c.csv, then
#   !python finetune.py RUN TAG      RUN in {b, c}, TAG in {sci, bge, e5, mpnet}
#   -> ft_{RUN}_{TAG}_oof.npy (3 reps x 2494, out-of-fold) and ft_{RUN}_{TAG}_test.npy (596); copy them to work/.
# Model: pretrained encoder + mean pooling + 4 sigmoid heads for P(y>=k), k=2..5; utility = mean of the 4 probabilities.
# Text: "VENUE YEAR: title". Trained on train labels only; test = mean over seeds of models fitted on all of train.
#   run b: folds_b, BCE loss, 5 epochs, lr 3e-5, 3 test seeds
#   run c: folds_c, same recipe, different seeds
# Encoders: allenai/scibert_scivocab_uncased (Beltagy et al., 2019), BAAI/bge-base-en-v1.5 (Xiao et al., 2023),
#           intfloat/e5-base-v2 (Wang et al., 2022), sentence-transformers/all-mpnet-base-v2 (Song et al., 2020).
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
RUN, TAG = sys.argv[1], sys.argv[2]
CFG = {"b": dict(folds="folds_b.csv", seed_off=0, test_seed=9000, n_test=3, epochs=5, lr=3e-5, rank=False),
       "c": dict(folds="folds_c.csv", seed_off=777, test_seed=9500, n_test=3, epochs=5, lr=3e-5, rank=False)}[RUN]
MODEL = {"bge": "BAAI/bge-base-en-v1.5", "e5": "intfloat/e5-base-v2", "sci": "allenai/scibert_scivocab_uncased",
         "mpnet": "sentence-transformers/all-mpnet-base-v2"}[TAG]
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); F = pd.read_csv(CFG["folds"]); assert (F.id == tr.id).all()
y = tr.Label.values; dev = "cuda"; t0 = time.time()
EPOCHS, LR, BS, MAXLEN, LLRD, W_RANK = CFG["epochs"], CFG["lr"], 32, 48, 0.9, 0.5
G = (y - 1) / 4.0
txt = lambda d: (d.venue.str.upper() + " " + d.year.astype(str) + ": " + d.title.fillna("")).tolist()
Xtr, Xte = txt(tr), txt(te); T = np.stack([(y >= k) for k in (2, 3, 4, 5)], 1).astype(np.float32)
tok = AutoTokenizer.from_pretrained(MODEL)


class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(MODEL); s.drop = nn.Dropout(0.1); s.head = nn.Linear(s.enc.config.hidden_size, 4)
    def forward(s, **b):
        h = s.enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        return s.head(s.drop((h * m).sum(1) / m.sum(1)))


enc = lambda t: tok(t, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")


def fit_predict(idx, pred, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev)
    if CFG["rank"]:                                   # layer-wise learning-rate decay
        nl = net.enc.config.num_hidden_layers; groups = [{"params": net.head.parameters(), "lr": LR}]
        for n, p in net.enc.named_parameters():
            depth = nl if ("pooler" in n) else (int(n.split("layer.")[1].split(".")[0]) + 1 if "layer." in n else 0)
            groups.append({"params": [p], "lr": LR * (LLRD ** (nl - depth))})
        opt = torch.optim.AdamW(groups, lr=LR, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(idx) / BS)); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(); lossf = nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(idx)
        for i in range(0, len(perm), BS):
            bi = perm[i:i + BS]; b = {k: v.to(dev) for k, v in enc([Xtr[j] for j in bi]).items()}
            with torch.cuda.amp.autocast(): logit = net(**b).float()
            loss = lossf(logit, torch.tensor(T[bi]).to(dev))
            if CFG["rank"]:                           # pairwise RankNet term on the expected gain within the batch
                u = torch.sigmoid(logit).mean(1); g = torch.tensor(G[bi], dtype=torch.float32).to(dev)
                d = g.unsqueeze(1) - g.unsqueeze(0); mask = (d > 0).float()
                rank = (nn.functional.softplus(-(u.unsqueeze(1) - u.unsqueeze(0)) * 8.0) * mask).sum() / mask.sum().clamp(min=1)
                loss = loss + W_RANK * rank
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(pred), 128):
            b = {k: v.to(dev) for k, v in enc(pred[i:i + 128]).items()}
            with torch.cuda.amp.autocast(): out.append(torch.sigmoid(net(**b).float()).mean(1).cpu().numpy())
    del net; torch.cuda.empty_cache(); return np.concatenate(out)


oof = np.zeros((3, len(tr)))
for r in range(3):
    f = F[f"fold{r}"].values
    for k in range(5):
        a, b = np.where(f != k)[0], np.where(f == k)[0]; oof[r, b] = fit_predict(a, [Xtr[j] for j in b], 1000 * r + 10 * k + CFG["seed_off"])
    np.save(f"ft_{RUN}_{TAG}_oof.npy", oof.astype(np.float32)); print(TAG, "rep", r, f"{time.time()-t0:.0f}s", flush=True)
test = np.mean([fit_predict(np.arange(len(tr)), Xte, CFG["test_seed"] + s) for s in range(CFG["n_test"])], 0)
np.save(f"ft_{RUN}_{TAG}_test.npy", test.astype(np.float32)); print("DONE", RUN, TAG, f"{time.time()-t0:.0f}s", flush=True)
