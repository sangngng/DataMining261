# CO3029 Data Mining Assignment - Personalized Research Paper Recommendation

Repository for the **CO3029 Data Mining (Semester 261)** course assignment at Ho Chi Minh City University of Technology (HCMUT).

- **Competition**: [Assignment Data Mining - 261 (Kaggle)](https://www.kaggle.com/competitions/assignment-data-mining-261/overview)
- **Host**: ThS. Vũ Ngọc Tú (`tuvn@hcmut.edu.vn`)

---

## 1. Project Overview & Architecture

The objective is to recommend and prioritize research papers for a specific researcher based on their personalized reading preferences.
- **Taught Core Methods Used**: 
  - **Classification**: GBDT Classifier (LightGBM) to filter uninteresting papers (true rating = 1).
  - **Regression**: GBDT Regressor (LightGBM) to score and rank worthwhile papers (ratings 2–5).
  - **Clustering**: K-Means clustering ($K=8$) on TF-IDF title embeddings to discover latent research topics.
- **Evaluation Metric**: Harmonic Mean of Relevance ($Q$) and Coverage ($R$):
  $$\text{Score} = \frac{2QR}{Q+R}$$

---

## 2. Directory Structure

```
DataMining/
├── data/                  # Kaggle input files (train.csv, test.csv, sample_submission.csv)
├── notebooks/             # Exploratory Data Analysis & experiments
├── src/
│   ├── metric.py          # Official competition metric calculation
│   ├── utils.py           # Data loading, random seed, submission verification
│   ├── features.py        # TF-IDF, K-Means clustering, and metadata features
│   ├── models.py          # Two-stage recommender architecture
│   └── evaluate.py        # 5-Fold Stratified Cross Validation & threshold sweep
├── submissions/           # Generated submission CSVs ready for Kaggle
├── requirements.txt       # Dependencies
├── main.py                # End-to-end execution script
└── README.md
```

---

## 3. Environment Setup & Requirements

- **Python**: `>= 3.10` (tested on Python 3.13 macOS ARM64)
- **Setup Virtual Environment**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## 4. Data Preparation

Place the following competition files inside the `data/` directory:
- `train.csv` (2,494 papers with metadata and `Label`)
- `test.csv` (596 candidate papers)
- `sample_submission.csv` (submission format template)

---

## 5. Reproducing Results & Submission

To run 5-Fold Cross Validation, optimize threshold $T$, and generate the final submission file:

```bash
python main.py --seed 42 --sub-name submission_two_stage.csv
```

The script will automatically:
1. Verify inputs and set random seed `42`.
2. Run Stratified 5-Fold Cross Validation.
3. Search for the optimal threshold $T^*$ on Out-Of-Fold predictions.
4. Retrain on the entire dataset and predict on `test.csv`.
5. Format, validate, and save the result to `submissions/submission_two_stage.csv`.

