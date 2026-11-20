# Kaggle "Assignment Data Mining - 261" (CO3029, HCMUT) — Team "hí ộ"

Personalised paper recommendation. For each of the 596 test papers we output a `priority`
(`0` = not recommended, larger = read earlier). The metric is the F-score of Relevance Q
(position-weighted gain, `g = (y-1)/4`, `w = 1/log2(r+1)`) and Coverage R (recall of papers with
label > 1), computed separately on two hidden 298-paper splits (Public / Private).

## Repository layout
| Path | Content |
|---|---|
| `source-code/` | **Pipeline A**: 23-learner stack, title-embedding learners, four fine-tuned encoders (`blend_ft4`). Details in `source-code/README.md`. |
| `pipeline-b/` | **Pipeline B / C**: an independent second implementation (own data reader, folds, learners, fine-tuned encoders). C = the same code re-run with other folds and seeds. |
| `pipeline-b/src/make_submission.py abc` | **Final submission**: equal rank average of A, B and C. |
| `reports/`, `pipeline-b/reports/` | EDA numbers and figures, CV tables. |
| `source-code/submissions/`, `pipeline-b/submissions/` | Every CSV submitted to Kaggle, byte-identical to what the code regenerates. |

## Method in one paragraph
Every pipeline turns the title / venue / year / authors of a paper into a utility score with models trained
on the 2,494 labelled papers, rank-normalises it, and recommends the top 65% (priorities 10 → 0.01, strictly
decreasing, the rest 0). The keep fraction was chosen on CV (flat between 0.60 and 0.65, worse from 0.70).
Pipelines A, B and C are trained on different folds and seeds and averaged with equal weights.

Taught approaches and where they enter the final priority:
- **Classification**: logistic P(y>1) and ordinal logistic P(y≥k) (A: base learners of the stack; B/C: `models_local.py`); LightGBM classifiers on embedding features; fine-tuned encoders are ordinal classifiers with four sigmoid heads.
- **Regression**: ridge and LightGBM regression of the gain `(y-1)/4` (A and B/C).
- **Clustering**: KMeans on TF-IDF/embedding vectors (A: k = 12, 40; B/C: k = 20), out-of-fold cluster target encoding and distance to centroid as features.
- **Itemset-based recommendation**: frequent 1-/2-itemsets of {title words, authors, venue, year bin}; association rules X → "worth reading" give a smoothed confidence and mean gain per paper (A: `assoc_*` learners; B/C: `rule_*` in `models_local.py`).

## Data and external resources (declared)
- Competition files: `pipeline-b/data/` holds the raw downloads (`train.csv`, `Test.csv`, `Submission.csv`);
  `source-code/data/` holds cleaned copies. `Test.csv` is not quoted, so commas inside titles split fields;
  both pipelines rebuild each row from fixed anchors (id, venue + year, DOI).
- Pretrained models (all run on Google Colab T4, outputs stored as `.npy` in the repo because they need a GPU):
  `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2019, Apache-2.0), `BAAI/bge-base-en-v1.5`
  (Xiao et al., 2023, MIT), `intfloat/e5-base-v2` (Wang et al., 2022, MIT), `allenai/scibert_scivocab_uncased`
  (Beltagy et al., 2019, Apache-2.0), `sentence-transformers/all-mpnet-base-v2` (Song et al., 2020, Apache-2.0).
  Fine-tuning uses train labels only.
- Pipeline A also uses abstracts fetched from OpenAlex, CrossRef and Semantic Scholar
  (`source-code/src/data_prep/fetch_*.py`, cached in `source-code/data/abstracts_cache.json`).
- **Unlabelled test metadata is used without labels**: A fits TF-IDF vocabulary, SVD and KMeans on train+test;
  B/C fit SVD and KMeans on train+test title embeddings. The CV follows the same protocol.
- No manual labels, no hand edits of predictions, no information derived from leaderboard feedback.

## Reproduce
Python 3.10, `pip install -r requirements.txt`. CPU only (2 cores, ~3 GB RAM) except the Colab steps, whose
outputs are already in the repo. Seeds are fixed in the code (A: 42 + repeat; B: 2026–2028; C: 3031–3033).
```bash
# Pipeline A (about 10 min)
cd source-code
python -m pytest tests                                    # metric unit tests
python -m src.eda                                         # reports/eda_summary.txt + figures
for r in 0 1 2 3 4; do python -m src.validate cache --rep $r; done
python -m src.validate report --reps 0 1 2                # reports/cv_results.csv, cv_summary.txt
python -m src.export_oof                                  # data/oof_blend_a.csv (OOF utilities for the comparison)
python -m src.make_submission --model blend_ft4 --keep 0.65 --dump-utility ../pipeline-b/work/utility_a_test.csv
# Pipeline B / C and the final average (about 5 min)
cd ../pipeline-b
python src/prep.py                                        # clean data + folds_b.csv, folds_c.csv
(cd src && python test_metric.py)
for r in b c; do python src/models_local.py $r; python src/models_emb.py $r; done
python src/compare.py                                     # reports/cv_compare.txt
python src/make_submission.py b; python src/make_submission.py ab; python src/make_submission.py abc
# Optional, GPU: colab/embed.py and colab/finetune.py RUN TAG regenerate work/emb_*.npy and work/ft_*.npy
```

## Validation protocol
Repeated 5-fold StratifiedGroupKFold; groups join exact-DOI duplicates and near-duplicate titles (TF-IDF cosine
≥ 0.8). Every label-dependent step (target encodings, stacking, rules, fine-tuning) is fitted inside the training
folds. Scores are computed with the official metric on random 298-paper subsets of the out-of-fold predictions whose
venue × year mix matches the test set (test has no ICLP papers and more 2016–2021 papers). The split-to-split
standard deviation is about 0.018, so differences below ~0.01 are within noise.

## CV results (`pipeline-b/reports/cv_compare.txt`, 2,500 simulated splits, same splits for every row)
| Model | keep | AUC(y>1) | Q | R | Score |
|---|---|---|---|---|---|
| A: blend_ft4 | 0.65 | 0.828 | 0.512 | 0.833 | 0.634 |
| B | 0.60 | 0.830 | 0.529 | 0.790 | 0.633 |
| C (B re-run) | 0.65 | 0.825 | 0.509 | 0.829 | 0.630 |
| avg A+B | 0.65 | 0.832 | 0.512 | 0.836 | 0.635 |
| **avg A+B+C (final)** | 0.65 | 0.831 | 0.513 | 0.837 | **0.636** |
| avg A+B+D (ranking-loss encoders) | 0.65 | 0.831 | 0.512 | 0.834 | 0.634 |

Earlier stages of pipeline A (its own protocol, `reports/cv_summary.txt` and `source-code/README.md`): random order
0.429, best single learner (char TF-IDF + LogReg) 0.583, 23-learner stack 0.602, stack + title-embedding learners
0.615, + four fine-tuned encoders 0.623.

## Submissions (in the order they were made)
| # | File in this repository | File name on Kaggle | Method | Public |
|---|---|---|---|---|
| 1 | `source-code/submissions/submission_stack18_keep65_387.csv` | `submission_v48_clean_stack_387.csv` | 18-learner stack, keep 0.65 | 0.63920 |
| 2 | `source-code/submissions/submission_stack23_keep75_447.csv` | `submission_v49_stack_keep75_447.csv` | 23-learner stack, keep 0.75 | 0.62747 |
| 3 | `source-code/submissions/submission_stack23_keep65_387.csv` | same | 23-learner stack, keep 0.65 | 0.63972 |
| 4 | `source-code/submissions/submission_blend23_keep65_387.csv` | same | stack + title-embedding learners | 0.64912 |
| 5 | `source-code/submissions/submission_blendft4_23_keep65_387.csv` | same | pipeline A (blend_ft4) | 0.66357 |
| 6 | `pipeline-b/submissions/submission_b_keep60_358.csv` | `submission_v2_keep60_358.csv` | pipeline B, keep 0.60 | 0.66497 |
| 7 | `pipeline-b/submissions/submission_avg_ab_keep65_387.csv` | `submission_avg_old_v2_keep65_387.csv` | average A+B | 0.66240 |
| 8 | `pipeline-b/submissions/submission_avg_abc_keep65_387.csv` | `submission_avg3_keep65_387.csv` | average A+B+C (final) | 0.65964 |

Kaggle file names differ for #1, #2 and #6–#8 because they were uploaded before the files were renamed here; the
contents are identical (same MD5). Other CSVs in the submission folders are keep-fraction candidates that were not
uploaded. The final submissions are chosen by CV, not by Public score.

## Notes and limitations
- The instructor's Public benchmark is 0.60953; all submissions above are higher.
- Before this repository the team made exploratory Kaggle submissions (V1–V47, branch `main`) that were tuned
  on the Public leaderboard; none of that code or output is used here.
- The Overview mentions coverage "through title similarity"; the Evaluation and Data pages define Coverage as
  recall of label > 1. We follow the Evaluation page.
- Tried without CV gain: more title encoders (SPECTER, GTE, BGE-large, E5-large, MiniLM-L12), title + abstract
  encoders, LambdaRank, learned stacking of the pipelines, venue-specific keep fractions, year re-weighting,
  pseudo-labels, ranking-loss fine-tuning (D). The remaining error is mostly the researcher's personal taste,
  which title, venue and year capture only partly.

## Access
Repository access is granted to **tuvn@hcmut.edu.vn**. Team members, student IDs and contributions: see the report.
