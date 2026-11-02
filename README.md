# CO3029 Data Mining Assignment - Personalized Research Paper Recommendation

[![Competition Status](https://img.shields.io/badge/Kaggle-Top%208%20(0.66762)-gold.svg)](https://www.kaggle.com/competitions/assignment-data-mining-261/leaderboard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repository for the **CO3029 Data Mining (Semester 261)** course assignment at **Ho Chi Minh City University of Technology (HCMUT)**.

- **Competition**: [Assignment Data Mining - 261 (Kaggle)](https://www.kaggle.com/competitions/assignment-data-mining-261/overview)
- **Course Instructor**: ThS. Vũ Ngọc Tú (`tuvn@hcmut.edu.vn`)
- **Team**: `hí ộ` (`catoxg`)
- **Team Members**:
  - Nguyễn Trọng Sáng (2353046) — **Team Lead, Re-ranking & Architecture**
  - Nguyễn Đức Vĩ (2353338) — **NLP, Scientific Transformer & Embeddings**
  - Đỗ Đình Duân (2352166) — **EDA, Feature Engineering & Academic Report**
- **Current Standing**: **Rank 8 / Top 8** — **`0.66762`** (Gap to Rank 7: `0.00038`, Gap to Top 2: `0.00242`)

---

## 1. Project Overview & Architectural Evolution

The objective is to recommend and prioritize research publications for a domain expert based on historical reading preferences ($y \in \{1, 2, 3, 4, 5\}$).
- **Core Taught Methods Applied**:
  1. **Clustering**: K-Means clustering ($K=12$) on dense Title+Abstract representations to discover latent research subfields (Answer Set Programming, Neural Network Verification, Description Logics, etc.).
  2. **Classification & Filtering**: Gradient Boosted Trees (LightGBM/XGBoost/CatBoost) and Scientific Transformers estimating $P(y \ge 2)$ to eliminate uninteresting publications ($y=1$).
  3. **Regression & Learning-to-Rank**: Probability Ranking Principle (PRP) optimizing continuous expected gain $\mathbb{E}[g]$ under non-linear DCG discounts.
  4. **Association Rule Mining**: Apriori frequent itemset mining on high-priority subsets.

- **Official Competition Metric**:
  $$\text{Score} = \frac{2 \cdot Q \cdot R}{Q + R}, \quad Q = \frac{\sum_{r=1}^{M} w_r \cdot g_r}{\sum_{r=1}^{M} w_r}, \quad w_r = \frac{1}{\log_2(r+1)}, \quad g_r = \frac{2^{y_r} - 1}{2^5 - 1}$$

---

## 2. Key Architectural Milestones

| Milestone | Core Technological Paradigm | CV Score | CV $Q$ | CV $R$ | Budget $M$ | Kaggle Public LB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V1 (Baseline)** | Two-Stage LightGBM + Sparse TF-IDF (2500) + KMeans (8 clusters) | $0.60431$ | $0.49595$ | $0.75801$ | $381$ | $0.61546$ |
| **V2 (SBERT)** | Sentence-BERT (MiniLM 384d) + Preference Centroids + GBDT Ensembling | $0.61256$ | $0.49175$ | $0.81207$ | $372$ | $0.65006$ |
| **V16 (Tri-Arch)** | MiniLM Core + 3 Diverse Inductive Paradigms (Topic + Centroids + k-NN) | $0.62339$ | $0.48955$ | $\mathbf{0.85795}$ | $381$ | $0.65474$ |
| **V20 (Grand Apex)** | Tri-Paradigm Decoupled Consensus + Frank & Hall Cumulative Ordinal Gain | $0.62910$ | $0.51150$ | $0.85820$ | $381$ | $0.65687$ |
| **V25B (Transformer)**| End-to-End Fine-Tuned Transformer (MPS, Pairwise Loss) + Budget Calibration | $0.59976$ | $0.49863$ | $0.75236$ | $365$ | $0.65947$ |
| **V26C (Topic Apex)** | Unsupervised K-Means ($k=12$) Discovery + Micro-Surgical Boundary Pruning | $0.60120$ | $0.49150$ | $0.77500$ | $365$ | $0.66297$ |
| **V28C (3-Engine Apex)**| Venue Asymmetry Recovery (Budget $365 \to 374$) + 82% Abstract PRP + Zero Ties | $0.61850$ | $0.53500$ | $0.81500$ | $374$ | $0.66753$ |
| **V29A (Diamond Apex)**| **Hyper Top-22 Ground-Truth Diamond Authors ($\mathbb{E}[g]$ Resonance) + Strict Convex Priority** | $\mathbf{0.61920}$ | $\mathbf{0.53800}$ | $0.81500$ | $374$ | $\mathbf{0.66762}$ (**All-Time Peak**) |

*Detailed experimental trajectory for all 35 configurations is comprehensively recorded in [`reports/experiment_log.csv`](reports/experiment_log.csv).*

---

## 3. Directory Structure

```
DataMining/
├── source-code/           # Core source code and model pipelines
│   ├── data/              # Dataset files (train.csv, test.csv, test_with_abstracts.csv)
│   ├── src/               # Library modules and scripts
│   │   ├── metric.py      # Official competition metric calculation
│   │   ├── utils.py       # Helper functions (seed, I/O, submission validation)
│   │   ├── features.py    # Feature extraction (TF-IDF, SBERT embeddings, Bayes priors)
│   │   ├── models.py      # Model architectures (GBDT, LambdaRank, Multi-stage)
│   │   ├── evaluate.py    # Stratified 5-Fold CV and threshold calibration
│   │   ├── create_v28c_breakthrough.py # V28C 3-Engine Breakthrough pipeline
│   │   ├── create_v29_apex.py          # V29 Diamond Resonance & Staged Escalation
│   │   └── create_v29c_pure_fusion.py  # V29C Pure Late-Fusion Apex pipeline
│   └── submissions/       # Evaluated Kaggle submission CSV files
├── reports/               # Experiment logs and analytical charts
│   ├── figures/           # EDA, clustering, and SHAP interpretability charts
│   └── experiment_log.csv # Exhaustive 35-run experimental logbook
├── report/                # Academic LaTeX report (Rubric: 10/10)
│   ├── chapters/          # Chapters 01 to 07
│   ├── figures/           # Embedded report figures
│   └── main.tex           # Main LaTeX document
└── README.md
```

---

## 4. Quickstart & Reproducibility

### Setup Virtual Environment:
```bash
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell
pip install -r source-code/requirements.txt
```

### Reproduce Key Submissions:
```bash
# 1. Generate V28C Breakthrough (Score: 0.66753)
python source-code/src/create_v28c_breakthrough.py

# 2. Generate V29A Diamond Resonance (Score: 0.66762 - All-Time Peak)
python source-code/src/create_v29_apex.py

# 3. Generate V29C Pure Late-Fusion Apex
python source-code/src/create_v29c_pure_fusion.py
```

All generated files undergo automated validation enforcing:
- Exactly 596 test rows.
- Exactly 374 active non-zero priorities.
- Exactly 374 unique values (**Strict Monotonic Convex Mapping, Zero Duplicate Ties**).
- Zero NaN / Inf values.


