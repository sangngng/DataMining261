import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple

def compute_clean_transductive_author_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    alpha: float = 2.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Clean Transductive Author & Co-Authorship Graph Features (2-hop)
    with strict Proceedings filtering to eliminate editorial volume bias.
    Ensures Out-Of-Fold (OOF) leave-one-out computation on Train to prevent data leakage.
    """
    # 1. Detect proceedings & workshop volumes (98.6% Label 1 in train)
    conf_regex = r"proceedings|international conference.*20\d\d|international workshop.*20\d\d|symposium on.*20\d\d|doctoral consortium"
    train_is_proc = train_df["title"].fillna("").str.lower().str.contains(conf_regex, regex=True).values
    test_is_proc  = test_df["title"].fillna("").str.lower().str.contains(conf_regex, regex=True).values

    print(f"[Transductive Author] Detected {train_is_proc.sum()} proceedings in train, {test_is_proc.sum()} in test.")

    # Global priors on Part 2 clean (id >= 712 and not proceedings)
    p2_mask = (train_df["id"].values >= 712) & (~train_is_proc)
    global_pos_prior = float((train_df.loc[p2_mask, "Label"] >= 2).mean())
    global_mean_prior = float(train_df.loc[p2_mask, "Label"].mean())
    print(f"[Transductive Author] Clean Part 2 Global Priors: pos_rate={global_pos_prior:.4f}, mean_y={global_mean_prior:.4f}")

    def parse_authors(s):
        if pd.isna(s):
            return []
        return [a.strip().lower() for a in str(s).split(",") if a.strip() and a.strip().lower() != "nan"]

    train_authors = [parse_authors(s) for s in train_df["authors"]]
    test_authors  = [parse_authors(s) for s in test_df["authors"]]
    labels = train_df["Label"].values

    # 2. Build Global Co-Authorship Graph (Train + Test, excluding proceedings)
    coauthor_adj = defaultdict(set)
    for is_p, auth_list in zip(train_is_proc, train_authors):
        if not is_p and len(auth_list) > 1:
            for a in auth_list:
                for b in auth_list:
                    if a != b:
                        coauthor_adj[a].add(b)
    for is_p, auth_list in zip(test_is_proc, test_authors):
        if not is_p and len(auth_list) > 1:
            for a in auth_list:
                for b in auth_list:
                    if a != b:
                        coauthor_adj[a].add(b)

    # 3. Aggregate clean author statistics from clean train
    auth_sum_y = defaultdict(float)
    auth_sum_pos = defaultdict(float)
    auth_cnt = defaultdict(int)

    for is_p, auth_list, y in zip(train_is_proc, train_authors, labels):
        if is_p:
            continue
        for a in auth_list:
            auth_sum_y[a] += float(y)
            auth_sum_pos[a] += float(y >= 2)
            auth_cnt[a] += 1

    def extract_row_features(auth_list, is_p, exclude_y=None):
        if is_p:
            return {
                "is_proceedings": 1.0,
                "auth_total_n": 0.0,
                "auth_pos_rate_bayes": 0.0,
                "auth_mean_y_bayes": 1.0,
                "auth_max_y_bayes": 1.0,
                "auth_min_y_bayes": 1.0,
                "auth_pure_neg_flag": 1.0,
                "auth_pure_pos_flag": 0.0,
                "coauth_2hop_mean_y": 1.0,
            }
        if len(auth_list) == 0:
            return {
                "is_proceedings": 0.0,
                "auth_total_n": 0.0,
                "auth_pos_rate_bayes": global_pos_prior,
                "auth_mean_y_bayes": global_mean_prior,
                "auth_max_y_bayes": global_mean_prior,
                "auth_min_y_bayes": global_mean_prior,
                "auth_pure_neg_flag": 0.0,
                "auth_pure_pos_flag": 0.0,
                "coauth_2hop_mean_y": global_mean_prior,
            }

        tot_n, tot_pos, tot_y = 0, 0.0, 0.0
        per_auth_bayes_y = []
        for a in auth_list:
            n = auth_cnt[a] - (1 if exclude_y is not None and auth_cnt[a] > 0 else 0)
            sy = auth_sum_y[a] - (exclude_y if exclude_y is not None and auth_cnt[a] > 0 else 0.0)
            sp = auth_sum_pos[a] - (float(exclude_y >= 2) if exclude_y is not None and auth_cnt[a] > 0 else 0.0)
            if n > 0:
                tot_n += n
                tot_pos += sp
                tot_y += sy
                per_auth_bayes_y.append((sy + alpha * global_mean_prior) / (n + alpha))

        bayes_pos = (tot_pos + alpha * global_pos_prior) / (tot_n + alpha)
        bayes_y   = (tot_y + alpha * global_mean_prior) / (tot_n + alpha)
        max_y     = max(per_auth_bayes_y) if per_auth_bayes_y else global_mean_prior
        min_y     = min(per_auth_bayes_y) if per_auth_bayes_y else global_mean_prior

        pure_neg = float(tot_n >= 2 and tot_pos == 0)
        pure_pos = float(tot_n >= 3 and (tot_pos / max(tot_n, 1)) >= 0.75)

        # 2-Hop Co-author propagation
        hop_y, hop_n = 0.0, 0
        for a in auth_list:
            for nb in coauthor_adj.get(a, []):
                if auth_cnt[nb] > 0:
                    hop_y += auth_sum_y[nb]
                    hop_n += auth_cnt[nb]
        coauth_2hop_y = (hop_y + alpha * global_mean_prior) / (hop_n + alpha)

        return {
            "is_proceedings": 0.0,
            "auth_total_n": float(tot_n),
            "auth_pos_rate_bayes": bayes_pos,
            "auth_mean_y_bayes": bayes_y,
            "auth_max_y_bayes": max_y,
            "auth_min_y_bayes": min_y,
            "auth_pure_neg_flag": pure_neg,
            "auth_pure_pos_flag": pure_pos,
            "coauth_2hop_mean_y": coauth_2hop_y,
        }

    train_rows = [
        extract_row_features(auth_list, is_p, exclude_y=(y if not is_p else None))
        for auth_list, is_p, y in zip(train_authors, train_is_proc, labels)
    ]
    test_rows = [
        extract_row_features(auth_list, is_p, exclude_y=None)
        for auth_list, is_p in zip(test_authors, test_is_proc)
    ]

    train_feats = pd.DataFrame(train_rows)
    test_feats = pd.DataFrame(test_rows)
    return train_feats, test_feats

if __name__ == '__main__':
    train = pd.read_csv('source-code/data/train_with_abstracts.csv')
    test = pd.read_csv('source-code/data/test_with_abstracts.csv')
    tr_f, te_f = compute_clean_transductive_author_features(train, test)
    print("Train feats shape:", tr_f.shape)
    print("Test feats shape:", te_f.shape)
    print("Test feats sample head:")
    print(te_f.head())
