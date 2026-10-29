"""
V24: Co-Authorship Community Empirical Bayes + Lexical/Syntactic Stacker + Harmonic Marginal Selector
Blueprint designed by Gemini Argon & Antigravity for breaking into Top 3 (>0.66303) and targeting Rank 1 (0.66962).

Run with:
PYTHONPATH=source-code .venv/bin/python source-code/src/train_v24_harmonic_community.py
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from src.metric import compute_competition_metric

DATA_DIR = Path("source-code/data")
SUB_DIR = Path("source-code/submissions")

def find_embedding_path(filename):
    for p in [Path("source-code/data") / filename, Path("data") / filename]:
        if p.exists():
            return p
    matches = list(Path(".").rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {filename}")


def parse_authors(author_str):
    if pd.isna(author_str) or not str(author_str).strip():
        return []
    parts = [re.sub(r"\s+", " ", a.strip().lower()) for a in str(author_str).split(",")]
    return [p for p in parts if len(p) > 2]


def extract_syntactic_and_domain_features(df):
    """
    Direction 6: Syntactic Title Decomposition & Domain Keyword Signals
    """
    titles = df["title"].fillna("").astype(str).tolist()
    venues = df["venue"].fillna("unknown").astype(str).str.lower().tolist()
    years = df["year"].fillna(2015.0).astype(float).values
    dois = df["doi"].fillna("").astype(str).tolist()
    authors_raw = df["authors"].fillna("").astype(str).tolist()

    CORE_POS_PATTERNS = [
        r"\banswer set\b", r"\basp\b", r"\bnonmonotonic\b", r"\blogic program",
        r"\bequilibrium\b", r"\bdisjunctive\b", r"\bepistemic\b", r"\bdefault logic\b",
        r"\babduct", r"\bargumentation\b", r"\bbelief revision\b", r"\bbelief change\b",
        r"\bdatalog\b", r"\bclingo\b", r"\bdlv\b", r"\bgringo\b", r"\bhex\b",
        r"\bjustification\b", r"\bexplanation\b", r"\bexplainable\b", r"\bplanning\b",
        r"\bmulti-shot\b", r"\bneural network verification\b", r"\bnnv\b", r"\bbackdoor\b",
        r"\bunfounded\b", r"\bstable model", r"\bwell-founded\b", r"\bcausal\b"
    ]
    MOD_POS_PATTERNS = [
        r"\bdescription logic", r"\bontology\b", r"\bmodal logic\b", r"\btemporal logic\b",
        r"\bfirst-order\b", r"\bhorn\b", r"\bsatisfiability\b", r"\bsat\b", r"\bsmt\b",
        r"\breachability\b", r"\bmodel checking\b", r"\bgame", r"\bstrategy\b"
    ]
    NEG_NOISE_PATTERNS = [
        r"\baircraft\b", r"\bavionic", r"\bswitching logic\b", r"\bhardware\b",
        r"\bcircuit\b", r"\bverilog\b", r"\bfpga\b", r"\bcompiler\b", r"\bcache\b",
        r"\boperating system\b", r"\bkernel\b", r"\bquantum\b", r"\bseparation logic\b",
        r"\bpointer\b", r"\bheap\b", r"\bmemory model\b", r"\bfloating-point\b",
        r"\bpacemaker\b", r"\bsimulink\b", r"\bhybrid automata\b"
    ]

    feats = []
    for t, v, y, d, a_str in zip(titles, venues, years, dois, authors_raw):
        t_low = t.lower()
        words = t.split()
        n_words = max(len(words), 1)
        n_chars = len(t)

        # Colon / System Tool syntax (e.g. "xASP: ...", "NNV 2.0: ...")
        has_colon = 1.0 if ":" in t else 0.0
        prefix_len = len(t.split(":")[0].split()) if ":" in t else 0.0
        is_tool_pattern = 1.0 if (":" in t and prefix_len <= 3) else 0.0

        # Acronym / Mixed-case solver token detection (e.g. xASP, clingo, DLV2, NNV)
        n_acronyms = sum(1 for w in words if (sum(1 for c in w if c.isupper()) >= 2 and len(w) >= 2))

        # Domain keyword counts
        core_hits = sum(1 for pat in CORE_POS_PATTERNS if re.search(pat, t_low))
        mod_hits  = sum(1 for pat in MOD_POS_PATTERNS if re.search(pat, t_low))
        neg_hits  = sum(1 for pat in NEG_NOISE_PATTERNS if re.search(pat, t_low))

        # Venue one-hot & interactions
        is_lpnmr = 1.0 if v == "lpnmr" else 0.0
        is_iclp  = 1.0 if v == "iclp" else 0.0
        is_kr    = 1.0 if v == "kr" else 0.0
        is_cav   = 1.0 if v == "cav" else 0.0
        is_lics  = 1.0 if v == "lics" else 0.0

        # Author count
        n_authors = len(parse_authors(a_str))
        has_doi = 1.0 if len(d.strip()) > 3 else 0.0

        feats.append([
            n_words, n_chars, has_colon, prefix_len, is_tool_pattern, n_acronyms,
            core_hits, mod_hits, neg_hits, core_hits - neg_hits,
            is_lpnmr, is_iclp, is_kr, is_cav, is_lics,
            is_cav * neg_hits, (is_lpnmr + is_iclp + is_kr) * (core_hits + 1.0),
            y, y - 2015.0, n_authors, has_doi
        ])
    return np.array(feats, dtype=np.float32)


def compute_oof_coauthorship_bayes_features(train_df, test_df, y_train, skf_splits):
    """
    Direction 4: 1-Hop & 2-Hop Co-Authorship Community Empirical Bayes Profiling (100% Leak-Free)
    Models research labs (Potsdam, TU Wien, Calabria, NMSU, etc.) via co-authorship graph.
    """
    all_df = pd.concat([train_df[["authors", "venue"]], test_df[["authors", "venue"]]], ignore_index=True)
    n_train = len(train_df)
    n_test = len(test_df)
    n_total = n_train + n_test

    author_lists = [parse_authors(s) for s in all_df["authors"]]
    venues = all_df["venue"].fillna("unknown").str.lower().tolist()

    # Build Co-Authorship Adjacency Graph across all 3,090 papers (unsupervised graph topology)
    coauthors = {}
    for alist in author_lists:
        for i in range(len(alist)):
            u = alist[i]
            if u not in coauthors:
                coauthors[u] = {}
            for j in range(len(alist)):
                if i != j:
                    v = alist[j]
                    coauthors[u][v] = coauthors[u].get(v, 0) + 1

    oof_auth_feats = np.zeros((n_train, 7), dtype=np.float32)
    test_auth_folds = np.zeros((n_test, 7), dtype=np.float32)

    gains_all = (y_train - 1.0) / 4.0
    w_all = (y_train >= 2).astype(np.float32)
    p45_all = (y_train >= 4).astype(np.float32)

    def _compute_for_split(tr_idx, eval_indices):
        # Venue priors in this fold
        venue_gain_prior = {}
        venue_w_prior = {}
        venue_p45_prior = {}
        global_gain = float(np.mean(gains_all[tr_idx]))
        global_w = float(np.mean(w_all[tr_idx]))
        global_p45 = float(np.mean(p45_all[tr_idx]))

        for v in ["lpnmr", "iclp", "kr", "cav", "lics"]:
            v_mask = [i for i in tr_idx if venues[i] == v]
            if len(v_mask) > 0:
                venue_gain_prior[v] = float(np.mean(gains_all[v_mask]))
                venue_w_prior[v] = float(np.mean(w_all[v_mask]))
                venue_p45_prior[v] = float(np.mean(p45_all[v_mask]))
            else:
                venue_gain_prior[v] = global_gain
                venue_w_prior[v] = global_w
                venue_p45_prior[v] = global_p45

        # 1-hop author stats strictly on tr_idx
        auth_n = {}
        auth_sum_gain = {}
        auth_sum_w = {}
        auth_sum_p45 = {}

        for i in tr_idx:
            g = gains_all[i]
            w = w_all[i]
            p45 = p45_all[i]
            for a in author_lists[i]:
                auth_n[a] = auth_n.get(a, 0.0) + 1.0
                auth_sum_gain[a] = auth_sum_gain.get(a, 0.0) + g
                auth_sum_w[a] = auth_sum_w.get(a, 0.0) + w
                auth_sum_p45[a] = auth_sum_p45.get(a, 0.0) + p45

        m_smooth = 5.5  # Strong Empirical Bayes shrinkage towards venue prior
        out = np.zeros((len(eval_indices), 7), dtype=np.float32)

        for row_k, idx in enumerate(eval_indices):
            v = venues[idx]
            pr_g = venue_gain_prior.get(v, global_gain)
            pr_w = venue_w_prior.get(v, global_w)
            pr_p45 = venue_p45_prior.get(v, global_p45)

            alist = author_lists[idx]
            if not alist:
                out[row_k] = [pr_g, pr_g, pr_w, pr_w, pr_p45, 0.0, 0.0]
                continue

            a_gains, a_ws, a_p45s = [], [], []
            total_1hop_support = 0.0
            total_2hop_support = 0.0

            for a in alist:
                n1 = auth_n.get(a, 0.0)
                sg1 = auth_sum_gain.get(a, 0.0)
                sw1 = auth_sum_w.get(a, 0.0)
                sp1 = auth_sum_p45.get(a, 0.0)

                # 2-hop co-author community aggregation
                n2, sg2, sw2, sp2 = 0.0, 0.0, 0.0, 0.0
                if a in coauthors:
                    for co_a, edge_w in coauthors[a].items():
                        if co_a in auth_n:
                            weight = min(float(edge_w), 3.0) * 0.25
                            n2  += weight * auth_n[co_a]
                            sg2 += weight * auth_sum_gain[co_a]
                            sw2 += weight * auth_sum_w[co_a]
                            sp2 += weight * auth_sum_p45[co_a]

                if n2 > 4.0:
                    scale = 4.0 / n2
                    n2, sg2, sw2, sp2 = n2 * scale, sg2 * scale, sw2 * scale, sp2 * scale

                eff_n = n1 + n2
                post_g = (sg1 + sg2 + m_smooth * pr_g) / (eff_n + m_smooth)
                post_w = (sw1 + sw2 + m_smooth * pr_w) / (eff_n + m_smooth)
                post_p45 = (sp1 + sp2 + m_smooth * pr_p45) / (eff_n + m_smooth)

                a_gains.append(post_g)
                a_ws.append(post_w)
                a_p45s.append(post_p45)
                total_1hop_support += n1
                total_2hop_support += n2

            out[row_k] = [
                float(np.max(a_gains)),
                float(np.mean(a_gains)),
                float(np.max(a_ws)),
                float(np.mean(a_ws)),
                float(np.max(a_p45s)),
                float(np.log1p(total_1hop_support)),
                float(np.log1p(total_2hop_support))
            ]
        return out

    test_indices = np.arange(n_train, n_total)
    for tr_idx, val_idx in skf_splits:
        oof_auth_feats[val_idx] = _compute_for_split(tr_idx, val_idx)
        test_auth_folds += _compute_for_split(tr_idx, test_indices) / len(skf_splits)

    return oof_auth_feats, test_auth_folds


def compute_oof_lexical_ridge_stacker(train_df, test_df, y_train, skf_splits):
    """
    Direction 2 & 6: Word + Char_wb TF-IDF N-gram Multi-Target Ridge Stacker
    """
    all_text = (
        pd.concat([train_df["venue"], test_df["venue"]], ignore_index=True).fillna("") + " " +
        pd.concat([train_df["title"], test_df["title"]], ignore_index=True).fillna("") + " " +
        pd.concat([train_df["authors"], test_df["authors"]], ignore_index=True).fillna("")
    ).tolist()

    vec_word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=12000, sublinear_tf=True)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=18000, sublinear_tf=True)

    X_word = vec_word.fit_transform(all_text)
    X_char = vec_char.fit_transform(all_text)

    X_lex = hstack([X_word, X_char]).tocsr()
    X_lex_tr = X_lex[:len(train_df)]
    X_lex_te = X_lex[len(train_df):]

    targets = [
        (y_train >= 2).astype(np.float32),
        (y_train >= 3).astype(np.float32),
        (y_train >= 4).astype(np.float32),
        (y_train == 5).astype(np.float32),
        (y_train - 1.0) / 4.0
    ]

    oof_lex = np.zeros((len(train_df), len(targets)), dtype=np.float32)
    test_lex = np.zeros((len(test_df), len(targets)), dtype=np.float32)

    for t_idx, y_t in enumerate(targets):
        for tr_idx, val_idx in skf_splits:
            model = Ridge(alpha=3.5, random_state=42)
            model.fit(X_lex_tr[tr_idx], y_t[tr_idx])
            oof_lex[val_idx, t_idx] = model.predict(X_lex_tr[val_idx])
            test_lex[:, t_idx] += model.predict(X_lex_te) / len(skf_splits)

    return oof_lex, test_lex


def compute_confidence_transductive_minilm_features(X_emb_tr, X_emb_te, y_train, skf_splits, v20_prio, v18_prio):
    """
    Direction 2 & 3: MiniLM Multi-Prototype Matching + Ultra-High-Confidence Pseudo-Labeled Anchors
    """
    pos_anchor_mask = (v20_prio >= np.percentile(v20_prio[v20_prio > 0], 80)) & (v18_prio > 0)
    neg_anchor_mask = (v20_prio == 0) & (v18_prio == 0)

    X_test_pos_anchors = X_emb_te[pos_anchor_mask]
    X_test_neg_anchors = X_emb_te[neg_anchor_mask]

    svd = TruncatedSVD(n_components=48, random_state=42)
    all_svd = svd.fit_transform(np.vstack([X_emb_tr, X_emb_te]))
    tr_svd, te_svd = all_svd[:len(X_emb_tr)], all_svd[len(X_emb_tr):]

    oof_sem = np.zeros((len(X_emb_tr), 10), dtype=np.float32)
    test_sem = np.zeros((len(X_emb_te), 10), dtype=np.float32)

    for tr_idx, val_idx in skf_splits:
        X_tr = X_emb_tr[tr_idx]
        y_tr = y_train[tr_idx]

        c_5 = 0.85 * X_tr[y_tr == 5].mean(axis=0, keepdims=True) + 0.15 * X_test_pos_anchors.mean(axis=0, keepdims=True)
        c_45 = X_tr[y_tr >= 4].mean(axis=0, keepdims=True)
        c_23 = X_tr[np.isin(y_tr, [2, 3])].mean(axis=0, keepdims=True)
        c_1  = 0.85 * X_tr[y_tr == 1].mean(axis=0, keepdims=True) + 0.15 * X_test_neg_anchors.mean(axis=0, keepdims=True)

        km_pos = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X_tr[y_tr >= 4])
        km_neg = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X_tr[y_tr == 1])

        def _extract_block(X_eval):
            s5  = cosine_similarity(X_eval, c_5).ravel()
            s45 = cosine_similarity(X_eval, c_45).ravel()
            s23 = cosine_similarity(X_eval, c_23).ravel()
            s1  = cosine_similarity(X_eval, c_1).ravel()

            max_proto_pos = cosine_similarity(X_eval, km_pos.cluster_centers_).max(axis=1)
            max_proto_neg = cosine_similarity(X_eval, km_neg.cluster_centers_).max(axis=1)

            sim_mat = cosine_similarity(X_eval, X_tr)
            top15 = np.argpartition(sim_mat, -15, axis=1)[:, -15:]
            rows = np.arange(len(X_eval))[:, None]
            w15 = np.clip(sim_mat[rows, top15], 0.01, 1.0) ** 2
            knn15_w = np.sum(w15 * (y_tr[top15] >= 2), axis=1) / np.sum(w15, axis=1)
            knn15_g = np.sum(w15 * ((y_tr[top15] - 1.0) / 4.0), axis=1) / np.sum(w15, axis=1)

            return np.column_stack([
                s5, s45, s23, s1,
                s45 - s1, s23 - s1,
                max_proto_pos, max_proto_pos - max_proto_neg,
                knn15_w, knn15_g
            ])

        oof_sem[val_idx] = _extract_block(X_emb_tr[val_idx])
        test_sem += _extract_block(X_emb_te) / len(skf_splits)

    return np.hstack([tr_svd, oof_sem]), np.hstack([te_svd, test_sem])


def main():
    print("="*75)
    print("=== LAUNCHING V24: HARMONIC-MARGINAL & CO-AUTHORSHIP COMMUNITY PIPELINE ===")
    print("="*75)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    y_train  = train_df["Label"].values.astype(int)

    # Load aligned V17, V18, V20 champions
    v17 = pd.read_csv(SUB_DIR / "submission_v17_decoupled_consensus.csv").set_index("id").reindex(test_df["id"]).reset_index()
    v18 = pd.read_csv(SUB_DIR / "submission_v18_transductive_consensus.csv").set_index("id").reindex(test_df["id"]).reset_index()
    v20 = pd.read_csv(SUB_DIR / "submission_v20_grand_apex.csv").set_index("id").reindex(test_df["id"]).reset_index()

    X_emb_tr = np.load(find_embedding_path("train_embeddings_minilm.npy"))
    X_emb_te = np.load(find_embedding_path("test_embeddings_minilm.npy"))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    skf_splits = list(skf.split(train_df, y_train))

    # 1. Multi-Modal Feature Extraction
    print("[1/4] Extracting Syntactic & Domain Features...")
    X_syn_tr = extract_syntactic_and_domain_features(train_df)
    X_syn_te = extract_syntactic_and_domain_features(test_df)

    print("[2/4] Computing 10-Fold Leak-Free Co-Authorship Community Empirical Bayes Features...")
    X_auth_tr, X_auth_te = compute_oof_coauthorship_bayes_features(train_df, test_df, y_train, skf_splits)

    print("[3/4] Computing 10-Fold OOF Word+Char TF-IDF Ridge Stacker...")
    X_lex_tr, X_lex_te = compute_oof_lexical_ridge_stacker(train_df, test_df, y_train, skf_splits)

    print("[4/4] Computing Prototype-Matched & Transductive MiniLM Features...")
    X_sem_tr, X_sem_te = compute_confidence_transductive_minilm_features(
        X_emb_tr, X_emb_te, y_train, skf_splits, v20["priority"].values, v18["priority"].values
    )

    X_tr_full = np.hstack([X_syn_tr, X_auth_tr, X_lex_tr, X_sem_tr])
    X_te_full = np.hstack([X_syn_te, X_auth_te, X_lex_te, X_sem_te])
    print(f"\n[FEATURE SUMMARY] Total V24 Feature Dimension: {X_tr_full.shape[1]} features")

    # 2. Train 4-Threshold Cumulative Ordinal Classifiers
    oof_P = np.zeros((len(train_df), 4), dtype=np.float64)   # P(y>=2), P(y>=3), P(y>=4), P(y==5)
    test_P = np.zeros((len(test_df), 4), dtype=np.float64)

    oof_pos_ranker = np.zeros(len(train_df), dtype=np.float64)
    test_pos_ranker = np.zeros(len(test_df), dtype=np.float64)

    print("\n[TRAINING] Fitting 10-Fold Cumulative Ordinal Classifiers (T2, T3, T4, T5)...")
    for k_idx, thresh in enumerate([2, 3, 4, 5]):
        y_bin = (y_train >= thresh).astype(int)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf_splits):
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=450, learning_rate=0.03, num_leaves=15,
                subsample_for_bin=1000, colsample_bytree=0.65, subsample=0.80,
                subsample_freq=1, min_child_samples=20, reg_alpha=1.5, reg_lambda=2.5,
                random_state=42 + k_idx, verbose=-1, n_jobs=-1
            )
            clf_lgb.fit(X_tr_full[tr_idx], y_bin[tr_idx])

            clf_cat = CatBoostClassifier(
                iterations=500, learning_rate=0.035, depth=5,
                l2_leaf_reg=5.0, random_seed=42 + k_idx, verbose=0, thread_count=-1
            )
            clf_cat.fit(X_tr_full[tr_idx], y_bin[tr_idx])

            p_val = 0.50 * clf_lgb.predict_proba(X_tr_full[val_idx])[:, 1] + 0.50 * clf_cat.predict_proba(X_tr_full[val_idx])[:, 1]
            p_te  = 0.50 * clf_lgb.predict_proba(X_te_full)[:, 1] + 0.50 * clf_cat.predict_proba(X_te_full)[:, 1]

            oof_P[val_idx, k_idx] = p_val
            test_P[:, k_idx] += p_te / len(skf_splits)
        print(f"  -> Threshold {thresh}: OOF Mean Probability = {oof_P[:, k_idx].mean():.4f}")

    print("\n[TRAINING] Fitting Positive-Only Specialist Ranker (y >= 2 subset with sample weighting)...")
    y_gain = (y_train - 1.0) / 4.0
    for fold_idx, (tr_idx, val_idx) in enumerate(skf_splits):
        pos_tr_idx = tr_idx[y_train[tr_idx] >= 2]
        sample_w = 1.0 + 1.5 * (y_gain[pos_tr_idx] ** 2)

        reg_cat = CatBoostRegressor(
            iterations=550, learning_rate=0.03, depth=5,
            l2_leaf_reg=6.0, random_seed=108, verbose=0, thread_count=-1
        )
        reg_cat.fit(X_tr_full[pos_tr_idx], y_gain[pos_tr_idx], sample_weight=sample_w)

        oof_pos_ranker[val_idx] = reg_cat.predict(X_tr_full[val_idx])
        test_pos_ranker += reg_cat.predict(X_te_full) / len(skf_splits)

    # 3. Compute Expected Gain & Harmonic Marginal Selection Score
    oof_E_gain  = 0.25 * np.sum(oof_P, axis=1)
    test_E_gain = 0.25 * np.sum(test_P, axis=1)

    # Stage 1 Harmonic Marginal Score: 0.35 * P(y>=2) + 0.65 * E[gain]
    oof_S_harm  = 0.35 * oof_P[:, 0] + 0.65 * oof_E_gain
    test_S_harm = 0.35 * test_P[:, 0] + 0.65 * test_E_gain

    # Stage 2 Convex Top-Heavy Ranking Score
    oof_Prio_v24  = 0.55 * (0.15*oof_P[:,0] + 0.20*oof_P[:,1] + 0.30*oof_P[:,2] + 0.35*oof_P[:,3]) + 0.45 * oof_pos_ranker
    test_Prio_v24 = 0.55 * (0.15*test_P[:,0] + 0.20*test_P[:,1] + 0.30*test_P[:,2] + 0.35*test_P[:,3]) + 0.45 * test_pos_ranker

    # Evaluate OOF CV across candidate budgets
    print("\n" + "="*75)
    print("=== OOF CROSS-VALIDATION SWEEP ON HARMONIC MARGINAL CRITERION ===")
    print("="*75)
    best_cv_score, best_q_pct = -1.0, 0.63914
    for q_pct in [0.620, 0.625, 0.630, 0.635, 0.63914, 0.645, 0.650]:
        thresh = np.quantile(oof_S_harm, 1.0 - q_pct)
        oof_sub_prio = np.where(oof_S_harm >= thresh, oof_Prio_v24 + 0.10, 0.0)
        score, Q, R = compute_competition_metric(train_df["id"].values, y_train, oof_sub_prio)
        recs_count = int((oof_sub_prio > 0).sum())
        print(f"[OOF CV] Budget={q_pct*100:.2f}% ({recs_count} recs) -> Score: {score:.5f} | Relevance Q: {Q:.5f} | Coverage R: {R:.5f}")
        if score > best_cv_score:
            best_cv_score = score
            best_q_pct = q_pct

    print(f"\n[PEAK CV] Best OOF Score: {best_cv_score:.5f} achieved at budget {best_q_pct*100:.2f}%")

    # Save OOF and Test Predictions
    oof_df = pd.DataFrame({
        "id": train_df["id"],
        "y_true": y_train,
        "p_ge2": oof_P[:, 0],
        "p_ge3": oof_P[:, 1],
        "p_ge4": oof_P[:, 2],
        "p_eq5": oof_P[:, 3],
        "e_gain": oof_E_gain,
        "s_harmonic": oof_S_harm,
        "prio_v24": oof_Prio_v24
    })
    oof_df.to_csv(SUB_DIR / "oof_v24_harmonic_community.csv", index=False)

    v24_pred_df = pd.DataFrame({
        "id": test_df["id"],
        "p_ge2": test_P[:, 0],
        "p_ge3": test_P[:, 1],
        "p_ge4": test_P[:, 2],
        "p_eq5": test_P[:, 3],
        "e_gain": test_E_gain,
        "s_harmonic": test_S_harm,
        "prio_v24": test_Prio_v24
    })
    v24_pred_df.to_csv(SUB_DIR / "test_v24_predictions.csv", index=False)

    # 4. Generate 3 Strategic Submissions for Today (29/10)
    print("\n" + "="*75)
    print("=== GENERATING 3 STRATEGIC SUBMISSIONS FOR 29/10 ===")
    print("="*75)
    r_v20_s1 = rankdata(v20["priority"].values) / len(test_df)
    r_v18_s1 = rankdata(v18["priority"].values) / len(test_df)
    r_v24_s1 = rankdata(test_S_harm) / len(test_df)

    # Fused Stage-1 Harmonic Selector (50% V20/V18 Anchor + 50% V24 Community-Lexical-Harmonic)
    fused_s1 = 0.30 * r_v20_s1 + 0.20 * r_v18_s1 + 0.50 * r_v24_s1

    # Stage-2 Ranking
    r_v20_s2 = rankdata(v20["priority"].values) / len(test_df)
    r_v24_s2 = rankdata(test_Prio_v24) / len(test_df)
    fused_s2 = 0.55 * r_v20_s2 + 0.45 * r_v24_s2

    candidates = [
        ("submission_v24_1_harmonic_381.csv", 381, fused_s1, fused_s2, "50% V20/V18 Anchor + 50% V24 Community-Lexical (Budget 381)"),
        ("submission_v24_2_precision_375.csv", 375, fused_s1, fused_s2, "Harmonic High-Precision (Budget 375) - Pruning lowest expected gain tail"),
        ("submission_v24_3_pure_v24_381.csv", 381, test_S_harm, test_Prio_v24, "Pure V24 Standalone Architecture (Budget 381)")
    ]

    for sub_name, budget, s1_score, s2_score, desc in candidates:
        top_idx = np.argsort(s1_score)[::-1][:budget]
        
        # Linear ranking mapping inside active budget to avoid ties and guarantee strictly positive priorities
        ranks = rankdata(s2_score[top_idx]) / float(budget)
        prio_scaled = 0.10 + 1.05 * (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
        
        prio_out = np.zeros(len(test_df), dtype=np.float64)
        prio_out[top_idx] = prio_scaled
        
        sub = pd.DataFrame({"id": test_df["id"], "priority": np.round(prio_out, 6)})
        out_file = SUB_DIR / sub_name
        sub.to_csv(out_file, index=False)
        
        overlap_v20 = int(((sub["priority"] > 0) & (v20["priority"] > 0)).sum())
        corr_v20 = float(sub["priority"].corr(v20["priority"]))
        print(f"[SAVED] {sub_name}")
        print(f"  Description: {desc}")
        print(f"  Active papers: {budget}/{len(test_df)} | Overlap with V20 (0.65687): {overlap_v20}/{budget}")
        print(f"  Correlation with V20: {corr_v20:.5f}")
        print()

    print("[SUCCESS] All V24 models trained and submissions generated successfully!")

if __name__ == "__main__":
    main()
