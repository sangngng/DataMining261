"""
train_v26_scientific_transformer.py: Abstract-Augmented Scientific Architecture (V26)
Formulated by Gemini Argon & Antigravity for breaking into Top 3 and targeting Rank 1 (0.68255).

Integrates:
  1. Full Head+Tail Abstract + Title Emphasis (Venue | Title.Title | Head+Tail Abstract)
  2. 10-Fold OOF Lexical TF-IDF Abstract Stacker (1-2 gram exact technical keywords)
  3. End-to-End Fine-Tuned Transformer on Apple Silicon MPS with Dual-Zone Pairwise Ranking Loss
  4. Part-2 Target Distribution Matching & Auxiliary Transfer Weighting on Part 1
  5. 75% V26 + 25% V25B/V20 Hybrid Ensemble at Golden Budgets (365, 368 recs)

Run with:
PYTHONPATH=source-code /Users/sanggnguyenn/Documents/HCMUT/DataMining/.venv/bin/python source-code/src/train_v26_scientific_transformer.py
"""
import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

from src.metric import compute_competition_metric

DATA_DIR = Path("source-code/data")
SUB_DIR  = Path("source-code/submissions")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Proven fast & stable geometry on Apple Silicon MPS

def build_full_scientific_text(df: pd.DataFrame) -> pd.Series:
    venues = df["venue"].fillna("unknown").astype(str).str.lower()
    titles = df["title"].fillna("").astype(str)
    abstracts = df["abstract"].fillna("").astype(str) if "abstract" in df.columns else pd.Series([""]*len(df))

    def format_row(v, t, a):
        words = str(a).split()
        if len(words) > 220:
            a_clean = " ".join(words[:150]) + " ... " + " ".join(words[-70:])
        elif len(words) > 0:
            a_clean = str(a)
        else:
            a_clean = t  # Fallback to title if abstract is empty
        return f"Venue: {v.upper()} | Title: {t} . {t} | Abstract: {a_clean}"

    return pd.Series([format_row(v, t, a) for v, t, a in zip(venues, titles, abstracts)], index=df.index)

def compute_lexical_abstract_oof_features(train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits: int = 10):
    """
    Extracts exact technical keyword signals (clingo, ASP, NNV, Marabou, Z3 vs hardware noise)
    from Title + Abstract via L2-regularized linear models over 1-2 gram TF-IDF.
    """
    print("[1/3] Computing Lexical Abstract TF-IDF Stacker...")
    train_text = build_full_scientific_text(train_df)
    test_text  = build_full_scientific_text(test_df)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        max_features=25000,
        strip_accents="unicode"
    )
    X_tr_tfidf = vec.fit_transform(train_text)
    X_te_tfidf = vec.transform(test_text)

    y = train_df["Label"].values.astype(int)
    # Sample weights: Part 2 (id >= 713) = 1.0, Part 1 (id < 713) = 0.35
    sample_weights = np.where(train_df["id"].values >= 713, 1.0, 0.35)

    oof_lexical = np.zeros((len(train_df), 5), dtype=np.float32)
    test_lexical = np.zeros((len(test_df), 5), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(train_df, y):
        sw_tr = sample_weights[tr_idx]
        for k_idx, thresh in enumerate([2, 3, 4, 5]):
            y_bin = (y[tr_idx] >= thresh).astype(int)
            clf = LogisticRegression(C=1.5, max_iter=500, random_state=42 + thresh)
            clf.fit(X_tr_tfidf[tr_idx], y_bin, sample_weight=sw_tr)
            oof_lexical[val_idx, k_idx] = clf.predict_proba(X_tr_tfidf[val_idx])[:, 1]
            test_lexical[:, k_idx] += clf.predict_proba(X_te_tfidf)[:, 1] / n_splits

        g_tr = (y[tr_idx] - 1.0) / 4.0
        reg = Ridge(alpha=3.0, random_state=42)
        reg.fit(X_tr_tfidf[tr_idx], g_tr, sample_weight=sw_tr)
        oof_lexical[val_idx, 4] = np.clip(reg.predict(X_tr_tfidf[val_idx]), 0.0, 1.0)
        test_lexical[:, 4] += np.clip(reg.predict(X_te_tfidf), 0.0, 1.0) / n_splits

    print(f"  -> Lexical Stacker Complete. Mean Stage-1 Prob (OOF): {oof_lexical[:, 0].mean():.4f}")
    return oof_lexical, test_lexical

class PaperAbstractDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None, weights=None, max_len=300):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
        )
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item

class ScientificRankerTransformer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.20)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size // 2, 5) # 4 ordinal logits + 1 gain head
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = torch.sum(out.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        logits = self.head(self.drop(pooled))
        return logits

def compute_dual_zone_ranking_loss(logits, labels, weights):
    device = logits.device
    ord_logits = logits[:, :4]
    gain_pred = torch.sigmoid(logits[:, 4])

    t2 = (labels >= 2).float()
    t3 = (labels >= 3).float()
    t4 = (labels >= 4).float()
    t5 = (labels == 5).float()
    ord_targets = torch.stack([t2, t3, t4, t5], dim=1)

    bce = nn.functional.binary_cross_entropy_with_logits(ord_logits, ord_targets, reduction="none")
    thresh_w = torch.tensor([1.3, 0.9, 1.0, 1.2], device=device).unsqueeze(0)
    loss_ord = (bce * thresh_w).mean(dim=1) * weights

    true_gain = (labels.float() - 1.0) / 4.0
    loss_gain = nn.functional.smooth_l1_loss(gain_pred, true_gain, reduction="none") * weights

    # Dual-Zone Pairwise Ranking:
    # Zone 1: Boundary Pairs (y_i >= 2 and y_j == 1)
    p2_pred = torch.sigmoid(ord_logits[:, 0])
    diff_p2 = p2_pred.unsqueeze(1) - p2_pred.unsqueeze(0)
    valid_bnd = ((labels.unsqueeze(1) >= 2) & (labels.unsqueeze(0) == 1)).float()
    loss_bnd = torch.relu(0.25 - diff_p2) * valid_bnd
    loss_bnd = loss_bnd.sum() / (valid_bnd.sum() + 1.0)

    # Zone 2: Top-DCG Pairs (y_i in {4, 5} and y_i > y_j)
    s_pred = 0.50 * gain_pred + 0.50 * torch.sigmoid(ord_logits).mean(dim=1)
    diff_true = true_gain.unsqueeze(1) - true_gain.unsqueeze(0)
    diff_s = s_pred.unsqueeze(1) - s_pred.unsqueeze(0)
    valid_top = ((diff_true > 0.05) & (labels.unsqueeze(1) >= 4)).float()
    loss_top = torch.relu(0.18 * diff_true - diff_s) * valid_top
    loss_top = loss_top.sum() / (valid_top.sum() + 1.0)

    return loss_ord.mean() + 0.80 * loss_gain.mean() + 0.50 * loss_bnd + 0.70 * loss_top

def predict_batch(model, loader, device):
    model.eval()
    all_p_sel, all_p_prio = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits[:, :4]).cpu().numpy()
            gain  = torch.sigmoid(logits[:, 4]).cpu().numpy()

            e_gain = 0.25 * probs.sum(axis=1)
            s_sel = 0.35 * probs[:, 0] + 0.65 * (0.5 * e_gain + 0.5 * gain)
            s_prio = 0.50 * (0.15*probs[:,0] + 0.20*probs[:,1] + 0.30*probs[:,2] + 0.35*probs[:,3]) + 0.50 * gain
            all_p_sel.append(s_sel)
            all_p_prio.append(s_prio)
    return np.concatenate(all_p_sel), np.concatenate(all_p_prio)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print("="*75)
    print(f"=== LAUNCHING V26: ABSTRACT-AUGMENTED SCIENTIFIC TRANSFORMER ON {device} ===")
    print("="*75)

    train_path = DATA_DIR / "train_with_abstracts.csv"
    test_path  = DATA_DIR / "test_with_abstracts.csv"
    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] Enriched files {train_path} not found. Run fetch_abstracts.py first!")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    print(f"Loaded train: {len(train_df)}, test: {len(test_df)}")

    # 1. Lexical Abstract Stacker
    oof_lex, te_lex = compute_lexical_abstract_oof_features(train_df, test_df)

    # 2. Transformer Fine-Tuning
    train_texts = build_full_scientific_text(train_df).tolist()
    test_texts  = build_full_scientific_text(test_df).tolist()
    tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)

    part1_idx = np.where(train_df["id"].values < 713)[0]
    part2_idx = np.where(train_df["id"].values >= 713)[0]
    y_all     = train_df["Label"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_sel_p2  = np.zeros(len(part2_idx), dtype=np.float64)
    oof_prio_p2 = np.zeros(len(part2_idx), dtype=np.float64)
    test_sel_ens = np.zeros(len(test_df), dtype=np.float64)
    test_prio_ens = np.zeros(len(test_df), dtype=np.float64)

    test_ds = PaperAbstractDataset(test_texts, tokenizer, max_len=300)
    test_loader = DataLoader(test_ds, batch_size=48, shuffle=False)

    print("\n[2/3] Fine-Tuning Transformer on Apple Silicon MPS (5 Folds)...")
    for fold, (p2_tr_sub, p2_val_sub) in enumerate(skf.split(part2_idx, y_all[part2_idx])):
        tr_indices = np.concatenate([part2_idx[p2_tr_sub], part1_idx])
        val_indices = part2_idx[p2_val_sub]

        tr_weights = np.ones(len(tr_indices), dtype=np.float32)
        for k, idx in enumerate(tr_indices):
            if idx < 713:
                tr_weights[k] = 0.25 if str(train_df.loc[idx, "venue"]).lower() == "iclp" else 0.40

        tr_ds = PaperAbstractDataset([train_texts[i] for i in tr_indices], tokenizer, y_all[tr_indices], tr_weights, max_len=300)
        val_ds = PaperAbstractDataset([train_texts[i] for i in val_indices], tokenizer, max_len=300)

        tr_loader = DataLoader(tr_ds, batch_size=24, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=48, shuffle=False)

        model = ScientificRankerTransformer(MODEL_NAME).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-5, weight_decay=0.015)
        epochs = 4
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(tr_loader)//2, num_training_steps=len(tr_loader)*epochs)

        best_val_sc = -1.0
        best_val_sel, best_val_prio = None, None
        best_te_sel, best_te_prio = None, None

        for ep in range(epochs):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad()
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["label"].to(device)
                wts  = batch["weight"].to(device)
                logits = model(ids, mask)
                loss = compute_dual_zone_ranking_loss(logits, lbls, wts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            v_sel, v_prio = predict_batch(model, val_loader, device)
            th = np.quantile(v_sel, 1.0 - (365.0 / 596.0))
            sub_p = np.where(v_sel >= th, v_prio + 0.1, 0.0)
            sc, Q, R = compute_competition_metric(train_df.loc[val_indices, "id"].values, y_all[val_indices], sub_p)
            if sc > best_val_sc:
                best_val_sc = sc
                best_val_sel, best_val_prio = v_sel, v_prio
                best_te_sel, best_te_prio = predict_batch(model, test_loader, device)

        oof_sel_p2[p2_val_sub] = best_val_sel
        oof_prio_p2[p2_val_sub] = best_val_prio
        test_sel_ens += best_te_sel / 5.0
        test_prio_ens += best_te_prio / 5.0
        print(f"  [Fold {fold+1}/5] Best Part-2 Val Score (@365 budget): {best_val_sc:.5f}")

    # Combine Transformer predictions with Lexical TF-IDF Stacker
    oof_lex_p2 = oof_lex[part2_idx]
    r_oof_lex_sel = rankdata(0.40*oof_lex_p2[:,0] + 0.60*oof_lex_p2[:,4]) / len(part2_idx)
    r_oof_trans_sel = rankdata(oof_sel_p2) / len(part2_idx)
    fused_oof_sel = 0.65 * r_oof_trans_sel + 0.35 * r_oof_lex_sel

    fused_oof_prio = 0.60 * oof_prio_p2 + 0.40 * oof_lex_p2[:, 4]

    print("\n" + "="*75)
    print("=== V26 PART-2 OOF VALIDATION ACROSS GOLDEN BUDGETS ===")
    print("="*75)
    for b_test in [355, 360, 365, 368, 372, 381]:
        th = np.quantile(fused_oof_sel, 1.0 - (b_test / 596.0))
        sub_p = np.where(fused_oof_sel >= th, fused_oof_prio + 0.10, 0.0)
        sc, Q, R = compute_competition_metric(train_df.loc[part2_idx, "id"].values, y_all[part2_idx], sub_p)
        print(f"  Budget={b_test}/596 ({b_test/596*100:.2f}%) -> Score: {sc:.5f} | Q: {Q:.5f} | R: {R:.5f}")

    # Save V26 standalone test predictions
    r_te_lex_sel = rankdata(0.40*te_lex[:,0] + 0.60*te_lex[:,4]) / len(test_df)
    r_te_trans_sel = rankdata(test_sel_ens) / len(test_df)
    v26_test_sel = 0.65 * r_te_trans_sel + 0.35 * r_te_lex_sel
    v26_test_prio = 0.60 * test_prio_ens + 0.40 * te_lex[:, 4]

    pd.DataFrame({
        "id": test_df["id"],
        "v26_sel": v26_test_sel,
        "v26_prio": v26_test_prio
    }).to_csv(SUB_DIR / "test_v26_abstract_predictions.csv", index=False)

    # 3. Hybrid Blending (75% V26 Abstract Transformer + 25% V25B/V20 Title Champions)
    print("\n[3/3] Generating Strategic V26 Hybrid Submissions...")
    v25b = pd.read_csv(SUB_DIR / "submission_v25b_transformer_hybrid_365.csv").set_index("id").reindex(test_df["id"]).reset_index()
    v20  = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test_df["id"]).reset_index()

    r_v25b_sel  = rankdata(v25b["priority"].values) / len(test_df)
    r_v20_sel   = rankdata(v20["priority"].values) / len(test_df)
    title_anchor_sel = 0.65 * r_v25b_sel + 0.35 * r_v20_sel

    # 75% V26 Abstract + 25% Title Anchor
    final_sel = 0.75 * v26_test_sel + 0.25 * title_anchor_sel
    final_prio = 0.70 * (rankdata(v26_test_prio) / len(test_df)) + 0.30 * (rankdata(v25b["priority"].values) / len(test_df))

    for b in [365, 368]:
        top_idx = np.argsort(final_sel)[::-1][:b]
        out_p = np.zeros(len(test_df), dtype=np.float64)
        out_p[top_idx] = final_prio[top_idx] - final_prio[top_idx].min() + 0.05
        sub = pd.DataFrame({"id": test_df["id"], "priority": np.round(out_p, 6)})
        out_name = f"submission_v26_abstract_hybrid_{b}.csv"
        sub.to_csv(SUB_DIR / out_name, index=False)
        overlap_v25b = ((sub["priority"] > 0) & (v25b["priority"] > 0)).sum()
        corr_v25b = sub["priority"].corr(v25b["priority"])
        print(f"[SAVED] {out_name} | Recommended: {b}/596 | Shared with V25B: {overlap_v25b}/{b} ({overlap_v25b/b*100:.1f}%) | Corr: {corr_v25b:.5f}")

    print("\n[SUCCESS] V26 Abstract-Augmented Pipeline Complete!")

if __name__ == "__main__":
    main()
