"""
Association Rule Mining on Paper-Title Keywords (Mục 4 - Insights, Rubric 1.5đ)

Mục tiêu: khai phá các "luật kết hợp" giữa các từ khóa xuất hiện trong tiêu đề
của những bài báo mà nhà nghiên cứu THỰC SỰ muốn đọc (Label >= 4), nhằm trả lời
câu hỏi "những chủ đề/kỹ thuật nào thường đi cùng nhau trong các bài báo ưu tiên
cao nhất?" — một insight định tính bổ sung cho pipeline dự đoán định lượng V3.

Pipeline:
  1. Lọc train.csv chỉ giữ các dòng có Label >= 4.
  2. Tokenize tiêu đề: lowercase, bỏ ký tự không phải chữ/số, loại stopword tiếng
     Anh (dùng sklearn.feature_extraction.text.ENGLISH_STOP_WORDS — đã có sẵn
     trong requirements.txt của repo, không cần cài thêm thư viện NLP mới), chỉ
     giữ token có độ dài >= 4 ký tự.
  3. Loại các từ khóa xuất hiện < 5 lần trong toàn bộ tập đã lọc (ngưỡng hỗ trợ
     tối thiểu ở cấp độ từ, trước khi mã hoá one-hot).
  4. Mã hoá mỗi bài báo thành một "giao dịch" (transaction) các từ khóa còn lại,
     dùng TransactionEncoder (mlxtend) để tạo ma trận one-hot.
  5. Khai phá luật kết hợp bằng FP-Growth (mlxtend.frequent_patterns.fpgrowth)
     với min_support=0.02, sau đó lọc luật association_rules với
     min_confidence=0.5 và min_lift=1.2.
  6. Xuất toàn bộ luật ra reports/association_rules.csv và in top 20 luật có
     lift cao nhất ra console.

Chạy: python notebooks/mine_association_rules.py (từ thư mục source-code/)
"""
import os
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

MIN_LABEL = 4            # Chỉ xét các bài "thực sự muốn đọc" (Label >= 4)
MIN_TOKEN_LEN = 4        # Độ dài tối thiểu của 1 từ khóa (ký tự)
MIN_WORD_COUNT = 5       # Từ khóa phải xuất hiện >= 5 lần trong tập đã lọc
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.2
TOP_N_PRINT = 20

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_title(title: str) -> list:
    """Lowercase + strip punctuation -> tokens, drop stopwords and short tokens."""
    if not isinstance(title, str):
        return []
    tokens = TOKEN_RE.findall(title.lower())
    return [t for t in tokens if len(t) >= MIN_TOKEN_LEN and t not in ENGLISH_STOP_WORDS]


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "train.csv")
    reports_dir = os.path.join(os.path.dirname(base_dir), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "association_rules.csv")

    df = pd.read_csv(data_path)
    print(f"Loaded train.csv: {df.shape[0]} rows")

    df_pref = df[df['Label'] >= MIN_LABEL].copy()
    print(f"Filtered to Label >= {MIN_LABEL}: {len(df_pref)} rows "
          f"({100.0 * len(df_pref) / len(df):.1f}% of train)")

    if len(df_pref) == 0:
        print("No rows with Label >= MIN_LABEL found. Aborting.")
        sys.exit(1)

    # 1. Tokenize each title into a keyword list (one "transaction" per paper)
    transactions_raw = df_pref['title'].apply(tokenize_title).tolist()

    # 2. Global word-frequency filter (>= MIN_WORD_COUNT occurrences across the filtered set)
    word_counts = {}
    for tokens in transactions_raw:
        for t in set(tokens):  # count each keyword once per paper (document frequency)
            word_counts[t] = word_counts.get(t, 0) + 1
    vocab = {w for w, c in word_counts.items() if c >= MIN_WORD_COUNT}
    print(f"Vocabulary after min_count={MIN_WORD_COUNT} filter: {len(vocab)} keywords "
          f"(out of {len(word_counts)} raw unique tokens)")

    transactions = [sorted(set(t for t in tokens if t in vocab)) for tokens in transactions_raw]
    transactions = [t for t in transactions if len(t) >= 2]  # need >= 2 items to form a rule
    print(f"Transactions with >= 2 qualifying keywords: {len(transactions)} / {len(transactions_raw)}")

    if len(transactions) == 0:
        print("No transactions left after filtering. Try lowering MIN_WORD_COUNT.")
        sys.exit(1)

    # 3. One-hot encode transactions
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    onehot_df = pd.DataFrame(te_array, columns=te.columns_)

    # 4. FP-Growth frequent itemsets
    frequent_itemsets = fpgrowth(onehot_df, min_support=MIN_SUPPORT, use_colnames=True)
    print(f"Frequent itemsets found (min_support={MIN_SUPPORT}): {len(frequent_itemsets)}")

    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found at this min_support. Try lowering MIN_SUPPORT.")
        sys.exit(1)

    # 5. Association rules filtered by confidence, then lift
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules['lift'] >= MIN_LIFT].copy()
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    print(f"Rules after confidence>={MIN_CONFIDENCE} and lift>={MIN_LIFT} filters: {len(rules)}")

    if len(rules) == 0:
        print("No rules survived the confidence/lift filters. Consider relaxing thresholds.")
        sys.exit(0)

    # Stringify frozensets for readable CSV output
    out_df = rules.copy()
    out_df['antecedents'] = out_df['antecedents'].apply(lambda s: ', '.join(sorted(s)))
    out_df['consequents'] = out_df['consequents'].apply(lambda s: ', '.join(sorted(s)))
    out_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    out_df = out_df[out_cols]
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rules to: {out_path}")

    print(f"\n===== TOP {TOP_N_PRINT} RULES BY LIFT =====")
    with pd.option_context('display.max_colwidth', 60, 'display.width', 140):
        print(out_df.head(TOP_N_PRINT).to_string(index=False))


if __name__ == "__main__":
    main()
