# Kaggle "Assignment Data Mining - 261" (CO3029, HCMUT) — Team "hí ộ"

Personalised paper recommendation: assign a `priority` to each of the 596 test papers
(`priority = 0` → not recommended). Metric = F-score of Relevance Q (position-weighted gain,
`g=(y-1)/4`, `w=1/log2(r+1)`) and Coverage R (recall of papers with label > 1), computed per
298-paper split.

## Approach (all learned directions: classification, regression, clustering, itemset rules)
Stacked ensemble of 23 base learners (out-of-fold predictions) → ridge + LightGBM meta-models →
utility = rank-average of regression and P(y>1) scores → keep top fraction (`--keep`), give strictly
decreasing priorities (10 → 0.01), set the rest to 0.

Base learners: word/char TF-IDF logistic & ridge, MiniLM embedding ridge/kNN/MLP, ordinal logistic
P(y≥k), author target-encoding, venue×year prior, KMeans (k=12, 40) target-encoding,
**Apriori-style itemset rules** (lift of single/pair items {title words, authors, venue, year} → y≥3),
SVD+GBDT, etc. See `source-code/src/models.py`.

## Data and external resources (declared)
- `source-code/data/{train,test,sample_submission}.csv`: official competition files.
- Abstracts (external, `train/test_with_abstracts.csv`): fetched from OpenAlex, CrossRef and
  Semantic Scholar APIs via `src/data_prep/fetch_*.py` (coverage 81% train / 71% test). Cached in
  `abstracts_cache.json`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2019, Apache-2.0),
  text = "VENUE: title. abstract[:300]"; stored in `clusters_k12.npz` (`src/data_prep/make_embeddings.py`).
- Title embeddings (`data/emb_{minilm,bge}_title_{train,test}.npy`): `all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5`
  (Xiao et al., 2023, MIT) applied to the **title only**, generated on Google Colab with
  `src/data_prep/make_specter_embeddings.py`; used by `src/embed_models.py` (kNN / prototype features + LightGBM).
- **Unlabelled test text is used unsupervisedly**: TF-IDF vocabulary, SVD, KMeans and item
  vocabulary are fitted on train+test (no labels). The CV mirrors this protocol.
- No manual labels, no hand edits of predictions, no Leaderboard-derived information in this pipeline.

## Reproduce
```bash
pip install -r requirements.txt
cd source-code
python -m pytest tests                          # metric unit tests (7)
python -m src.eda                               # reports/eda_summary.txt + figures
for r in 0 1 2; do python -m src.validate cache --rep $r; done   # ~90 s per repeat, 2 CPU
python -m src.validate report --reps 0 1 2      # reports/cv_results.csv, cv_summary.txt
python -m src.validate compare --reps 3 4       # 18 vs 23 base learners on fresh seeds
python -m src.make_submission --keep 0.60 0.65 0.70   # ~25 s, 23-learner stack only
python -m src.make_submission --model blend --keep 0.65   # final: stack + MiniLM-title + BGE-title learners (equal rank average)
```
Seeds: `SEED=42`, CV repeat r uses seed 42+r. Hardware used: 2 CPU, 3 GB RAM, Python 3.10.12.

## Validation protocol
Repeated 5-fold StratifiedGroupKFold (groups = near-duplicate titles, TF-IDF cosine ≥ 0.8; 3 dev repeats,
2 fresh repeats). Stacker inputs are nested inner-OOF predictions. Score is evaluated on random
298-paper subsets of the OOF predictions with the **test venue mix** (cav 37%, lics 31%, kr 25%, lpnmr 6%,
no iclp). Split-to-split std ≈ 0.02, so differences below ~0.01 are noise.

## Results (CV, `reports/cv_summary.txt`)
| Model | keep | Score | Q | R |
|---|---|---|---|---|
| Random order | 0.65 | 0.429 | 0.320 | 0.655 |
| Char TF-IDF + LogReg | 0.65 | 0.583 | 0.464 | 0.784 |
| Ordinal LogReg | 0.65 | 0.592 | 0.477 | 0.781 |
| Stack (23 learners) | 0.65 | 0.602 ± 0.021 | 0.485 | 0.794 |
| **Blend: stack + title-embedding learners (final)** | 0.65 | **0.615** | 0.496 | 0.813 |

The blend (5 CV repeats, AUC of y>1: 0.788 -> 0.807) improves the stack by +0.012 (paired 95% CI [-0.006, +0.030], P(>0)=0.9), with equal untuned weights. The stack beats the best simple baseline by +0.021 (paired 95% CI [+0.003, +0.041]). Using 23 instead of
18 base learners adds only +0.004 (CI includes 0). Text signal carries most of the gain
(without text: 0.566); authors / clusters / itemsets / venue-year add ≈0. Keep fraction is flat for 0.60–0.70.

## Selected submissions (`source-code/submissions/`)
| File | Method | Kaggle Public |
|---|---|---|
| `submission_v48_clean_stack_387.csv` | 18-learner stack, keep 0.65 | 0.63920 |
| `submission_v49_stack_keep75_447.csv` | 23-learner stack, keep 0.75 | 0.62747 |
| `submission_stack23_keep65_387.csv` | 23-learner stack, keep 0.65 | 0.63972 |
| `submission_blend23_keep65_387.csv` | stack + MiniLM-title + BGE-title learners, keep 0.65 | 0.64912 |
| `submission_stack23_keep{60,70}_*.csv`, `submission_blend23_keep70_417.csv` | other keep fractions (candidates) | not submitted |

Final 3 are chosen by CV (not Public). Submission IDs and the final commit hash are recorded in the report.

## Honest notes / limitations
- Clean Public scores (≈0.63–0.65) are **below** the team's earlier best (0.66862) and the baseline row
  (0.66962). That earlier history (branch `main`, V1–V47) contained manual ID adjustments and
  Public-guided blending; it is not used, not reproducible from code, and is excluded from this branch.
- The Overview text mentions coverage via "title similarity"; the Evaluation/Data pages define
  Coverage as recall of label > 1. We use the latter.
- CV ceiling for these features ≈ 0.60; the Public/Private gap shows distribution differences (label shift)
  that CV with train-only labels cannot capture. Single researcher's labels; ids correlate with venue order.
- Tried and rejected: LambdaRank, id-neighbour features, structural flags, further hyperparameter tuning.

## Access
Repository access is granted to **tuvn@hcmut.edu.vn**. Team members, student IDs and contributions: see the report.
