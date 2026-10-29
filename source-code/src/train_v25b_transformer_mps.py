"""
V25B: End-to-End Fine-Tuned Transformer on Apple Silicon MPS (Ordinal + Pairwise Ranking Loss)
Trained with Part-2 Target Distribution Matching & DOI/Author/Venue Self-Attention.
Run with: /Users/sanggnguyenn/Documents/HCMUT/DataMining/.venv/bin/python source-code/src/train_v25b_transformer_mps.py
"""
import os
import sys
sys.path.append("source-code")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

from src.metric import compute_competition_metric

DATA_DIR = Path("source-code/data")
SUB_DIR = Path("source-code/submissions")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Cached locally, fast & proven geometry on MPS

def get_doi_category(doi_str):
    d = str(doi_str).lower().strip()
    if "10.24963" in d:
        return "IJCAI_KR_Proceedings"
    if "ceur" in d:
        return "CEUR_Workshop"
    if "10.1007" in d:
        return "Springer_LNCS"
    if "10.1109" in d:
        return "IEEE_Proceedings"
    if "semanticscholar" in d:
        return "SemanticScholar_Archive"
    if d.startswith("10."):
        return "Standard_DOI"
    return "No_DOI"

def build_academic_prompt(df):
    prompts = []
    for _, row in df.iterrows():
        venue = str(row.get("venue", "unknown")).upper()
        year = int(float(row.get("year", 2015))) if not pd.isna(row.get("year")) else 2015
        doi_cat = get_doi_category(row.get("doi", ""))
        authors = str(row.get("authors", "Unknown"))
        title = str(row.get("title", ""))
        text = f"Venue: {venue} | Source: {doi_cat} | Year: {year} | Authors: {authors} | Title: {title}"
        prompts.append(text)
    return prompts

class PaperDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None, weights=None, max_len=160):
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

class AcademicRankerTransformer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.20)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size // 2, 5)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = torch.sum(out.last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        logits = self.head(self.drop(pooled))
        return logits

def compute_multitask_ranking_loss(logits, labels, weights):
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

    s_pred = 0.50 * gain_pred + 0.50 * torch.sigmoid(ord_logits).mean(dim=1)
    diff_true = true_gain.unsqueeze(1) - true_gain.unsqueeze(0)
    diff_pred = s_pred.unsqueeze(1) - s_pred.unsqueeze(0)
    valid_pairs = (diff_true > 0.05).float()
    margin = 0.15 * diff_true
    pair_loss_mat = torch.relu(margin - diff_pred) * valid_pairs
    w_pair = torch.sqrt(weights.unsqueeze(1) * weights.unsqueeze(0))
    loss_pair = (pair_loss_mat * w_pair).sum() / (valid_pairs.sum() + 1.0)

    return loss_ord.mean() + 0.80 * loss_gain.mean() + 1.20 * loss_pair

def predict_Scores(model, loader, device):
    model.eval()
    all_p_sel = []
    all_p_prio = []
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== V25B END-TO-END TRANSFORMER RANKER ON DEVICE: {device} ===")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")

    train_texts = build_academic_prompt(train_df)
    test_texts  = build_academic_prompt(test_df)
    tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)

    part1_idx = np.where(train_df["id"].values < 713)[0]
    part2_idx = np.where(train_df["id"].values >= 713)[0]
    y_all = train_df["Label"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_sel_part2 = np.zeros(len(part2_idx), dtype=np.float64)
    oof_prio_part2 = np.zeros(len(part2_idx), dtype=np.float64)
    test_sel_ens = np.zeros(len(test_df), dtype=np.float64)
    test_prio_ens = np.zeros(len(test_df), dtype=np.float64)

    test_ds = PaperDataset(test_texts, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("\n[TRAINING] 5-Fold Stratified Fine-Tuning strictly on Part 2 (with Part 1 downweighted)...")
    for fold, (p2_tr_sub, p2_val_sub) in enumerate(skf.split(part2_idx, y_all[part2_idx])):
        tr_indices = np.concatenate([part2_idx[p2_tr_sub], part1_idx])
        val_indices = part2_idx[p2_val_sub]

        tr_weights = np.ones(len(tr_indices), dtype=np.float32)
        for k, idx in enumerate(tr_indices):
            if idx < 713:
                tr_weights[k] = 0.25 if str(train_df.loc[idx, "venue"]).lower() == "iclp" else 0.45

        tr_ds = PaperDataset([train_texts[i] for i in tr_indices], tokenizer, y_all[tr_indices], tr_weights)
        val_ds = PaperDataset([train_texts[i] for i in val_indices], tokenizer)

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = AcademicRankerTransformer(MODEL_NAME).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.02)
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
                loss = compute_multitask_ranking_loss(logits, lbls, wts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            v_sel, v_prio = predict_Scores(model, val_loader, device)
            th = np.quantile(v_sel, 1.0 - (342.0 / 596.0))
            sub_p = np.where(v_sel >= th, v_prio + 0.1, 0.0)
            sc, Q, R = compute_competition_metric(train_df.loc[val_indices, "id"].values, y_all[val_indices], sub_p)
            if sc > best_val_sc:
                best_val_sc = sc
                best_val_sel, best_val_prio = v_sel, v_prio
                best_te_sel, best_te_prio = predict_Scores(model, test_loader, device)

        oof_sel_part2[p2_val_sub] = best_val_sel
        oof_prio_part2[p2_val_sub] = best_val_prio
        test_sel_ens += best_te_sel / 5.0
        test_prio_ens += best_te_prio / 5.0
        print(f"  [Fold {fold+1}/5] Best Part-2 Val Score (@57.4% budget): {best_val_sc:.5f}")

    print("\n" + "="*75)
    print("=== FULL PART-2 OOF EVALUATION ACROSS BUDGETS ===")
    print("="*75)
    for b_test in [335, 340, 342, 348, 354, 381]:
        th = np.quantile(oof_sel_part2, 1.0 - (b_test / 596.0))
        sub_p = np.where(oof_sel_part2 >= th, oof_prio_part2 + 0.1, 0.0)
        sc, Q, R = compute_competition_metric(train_df.loc[part2_idx, "id"].values, y_all[part2_idx], sub_p)
        print(f"[V25B Part-2 OOF] Budget={b_test}/596 -> Score: {sc:.5f} | Q: {Q:.5f} | R: {R:.5f}")

    pd.DataFrame({
        "id": test_df["id"],
        "v25b_sel": test_sel_ens,
        "v25b_prio": test_prio_ens
    }).to_csv(SUB_DIR / "test_v25b_transformer_predictions.csv", index=False)

    # Hybrid blend: Transformer V25B + V25A + V20
    v25a = pd.read_csv(SUB_DIR / "submission_v25a_part2_budget_342.csv").set_index("id").reindex(test_df["id"]).reset_index()
    v20  = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test_df["id"]).reset_index()

    r_trans_sel = rankdata(test_sel_ens) / len(test_df)
    r_v20_sel   = rankdata(v20["priority"].values) / len(test_df)
    r_v25a_sel  = rankdata(v25a["priority"].values) / len(test_df)

    final_sel = 0.45 * r_trans_sel + 0.35 * r_v25a_sel + 0.20 * r_v20_sel
    final_prio = 0.45 * (rankdata(test_prio_ens) / len(test_df)) + 0.55 * (rankdata(v20["priority"].values) / len(test_df))

    for target_budget in [342, 348]:
        top_idx = np.argsort(final_sel)[::-1][:target_budget]
        out_p = np.zeros(len(test_df), dtype=np.float64)
        out_p[top_idx] = final_prio[top_idx] - final_prio[top_idx].min() + 0.05
        sub = pd.DataFrame({"id": test_df["id"], "priority": np.round(out_p, 6)})
        out_name = f"submission_v25b_transformer_hybrid_{target_budget}.csv"
        sub.to_csv(SUB_DIR / out_name, index=False)
        overlap_v20 = ((sub["priority"] > 0) & (v20["priority"] > 0)).sum()
        corr_v20 = sub["priority"].corr(v20["priority"])
        print(f"\n[SAVED] {out_name} | Recommended: {target_budget}/596 | Shared with V20: {overlap_v20}/{target_budget} ({overlap_v20/target_budget*100:.1f}%) | Corr with V20: {corr_v20:.5f}")

if __name__ == "__main__":
    main()
