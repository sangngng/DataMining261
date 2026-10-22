# 🛰️ TASK_SYNC.md — Multi-Agent Collaboration Plan

> **Tài liệu phối hợp kiến trúc giữa Agent 1 (Antigravity - System Architect) và Agent 2 (Cline / Claude Code - Implementation & Execution Engineer).**
> **Ngày lập:** 22/10/2026  
> **Dự án:** CO3029 Data Mining (HK241) — Personalized Research Paper Recommendation  
> **Cuộc thi:** [Assignment Data Mining - 261 (Kaggle)](https://www.kaggle.com/competitions/assignment-data-mining-261)  
> **Đội thi:** `hí ộ` | **Vị trí hiện tại:** **Rank 5 / 16 (Score: 0.65006, 2 entries)**  
> **Mục tiêu chặng tới:** Bứt phá vào **TOP 1 (> 0.66962)** và hoàn thiện các yêu cầu thuật toán trong Rubric Báo cáo 10đ.

---

## 0. Mô Hình Quản Trị & Phân Vai 2 AI Agent (Governance Protocol)

Dự án được vận hành theo mô hình phân tầng chặt chẽ giữa 2 AI Agent:

```
┌────────────────────────────────────────────────────────┐
│        ANTIGRAVITY (Project Manager & Chief Architect) │
│  - Quản lý toàn bộ tiến độ dự án, GitHub, Kaggle       │
│  - Định hướng chiến lược, phân tích kiến trúc          │
│  - Giám sát Rubric 10đ và chất lượng sản phẩm bàn giao │
└───────────────────────────┬────────────────────────────┘
                            │ (Giao đề bài qua TASK_SYNC.md)
                            ▼
┌────────────────────────────────────────────────────────┐
│           CLINE / CLAUDE CODE (Execution Engineer)     │
│  - Nhận nhiệm vụ từ TASK_SYNC.md                       │
│  - Viết code chi tiết trong source-code/               │
│  - Chạy lệnh terminal, train mô hình, xuất submission   │
└────────────────────────────────────────────────────────┘
```

* **Quy tắc phối hợp:** 
  1. Cline tập trung vào việc hiện thực hóa code và thực thi lệnh theo đúng đặc tả của Kiến trúc sư (Antigravity).
  2. Mọi thay đổi về cấu trúc thư mục, kiến trúc mô hình hoặc chiến thuật nộp bài đều do Antigravity điều phối.
  3. Sau khi Cline hoàn thành một sprint, Antigravity sẽ kiểm tra tính đúng đắn, rà soát data leakage, đối chiếu Leaderboard Kaggle và cập nhật nhật ký thực nghiệm.

---

## 1. Bối cảnh & Hiện trạng Kỹ thuật (System Context)

### 1.1. Công thức Metric Cuộc thi:
$$\text{Score} = \frac{2 \cdot Q \cdot R}{Q + R}$$
* **Coverage ($R$):** Tỷ lệ bài nhãn 2–5 được gán `priority > 0`. Nếu lọc bỏ quá nhiều bài, $R$ tụt.
* **Relevance ($Q$):** Điểm chiết khấu DCG theo vị trí xếp hạng $r = 1, \dots, m$:
  $$Q = \frac{\sum_{r=1}^m w_r \cdot g_r}{\sum_{r=1}^m w_r}, \quad w_r = \frac{1}{\log_2(r + 1)}, \quad g_r = \frac{y_r - 1}{4}$$
  *Hệ quả kiến trúc:* Trọng số $w_r$ cực lớn ở các vị trí đầu ($r = 1 \to 30$). Đưa được bài nhãn 5 lên đầu và triệt tiêu bài nhãn 1 (gán `priority = 0.0`) là chìa khóa quyết định điểm số.

### 1.2. Lịch sử Thực nghiệm:
* **V1 (Baseline):** TF-IDF (2500) + KMeans(8) + Two-Stage LightGBM $\to$ CV: **0.60431**, Public LB: **0.61546** (Rank 13).
* **V2 (SBERT + GBDT Ensemble):** Sentence-BERT (`all-MiniLM-L6-v2`) + SVD(48) + KMeans(12) + Preference Centroids + LightGBM & XGBoost Ensemble $\to$ CV: **0.61601** (+0.0117), Public LB: **0.65006** (Rank 5).
* **Khoảng cách tới Top 1 (`KFC xin chao`: 0.66962):** Chỉ còn **0.0195 điểm**.

### 1.3. Môi trường Thực thi (Cline chú ý):
* Python: `>= 3.10` (máy đang chạy Python 3.13 macOS ARM64).
* Virtualenv: `.venv/` (đã kích hoạt sẵn `lightgbm`, `xgboost`, `scikit-learn`, `sentence-transformers`, `torch`).
* Thư mục mã nguồn chung: `source-code/`
* Dữ liệu: `source-code/data/train.csv`, `source-code/data/test.csv`.
* Cache Embeddings đã có: `source-code/data/train_embeddings_minilm.npy` (2494, 384), `source-code/data/test_embeddings_minilm.npy` (596, 384).

---

## 2. Mục tiêu Sprint V3: Đánh chiếm Top 1 (> 0.66962)

Agent 2 (Cline) cần triển khai 4 hạng mục kỹ thuật cốt lõi sau:

### 🎯 Mục tiêu 1: Out-of-Fold Author & Venue Target Encoding (High Leverage)
* **Vấn đề:** Hiện tại mô hình chỉ dùng `author_count`. Trong khi đó, các tác giả ruột (gu đọc của nhà nghiên cứu) thường xuyên được chấm 4–5 sao ở nhiều bài báo khác nhau.
* **Logic thuật toán:**
  1. Với mỗi bài báo, phân tách chuỗi `authors` (phân tách bởi dấu phẩy).
  2. Tính toán lịch sử của từng tác giả trên tập huấn luyện:
     - `author_max_rating`: Điểm đánh giá cao nhất trong quá khứ của tác giả.
     - `author_mean_rating`: Điểm đánh giá trung bình.
     - `author_paper_count`: Số lượng bài báo tác giả đã xuất bản.
     - `author_is_preferred`: Cờ 1 nếu tác giả từng có bài $\ge 4$ sao.
  3. Khi một bài báo có nhiều đồng tác giả, lấy `max()` và `mean()` trên danh sách tác giả của bài đó.
  4. **NGUYÊN TẮC BẮT BUỘC:** Trong quá trình 5-Fold Cross Validation, việc thống kê tác giả **chỉ được tính trên Train Fold**, sau đó map sang Validation Fold. Tuyệt đối không tính trên toàn bộ dữ liệu trước khi split để tránh rò rỉ dữ liệu (*Data Leakage*).
  5. Áp dụng Bayesian Smoothing cho Venue:
     $$S_{venue} = \frac{n \cdot \bar{y}_{venue} + m \cdot \bar{y}_{global}}{n + m} \quad (\text{với } m = 5)$$

### 🎯 Mục tiêu 2: Thay đổi Hàm Mục tiêu sang Ranking / LambdaRank
* **Vấn đề:** Mô hình Stage 2 hiện dùng MSE Regressor (coi sai số ở rank 1 giống sai số ở rank 300). Nhưng metric cuộc thi là Discounted Gain $w_r = 1/\log_2(r+1)$.
* **Logic thuật toán:**
  * Thử nghiệm mô hình chuyên xếp hạng: `lightgbm.LGBMRanker(objective="lambdarank")` hoặc hàm ranking kết hợp:
    $$\text{Priority} = \text{PredictedScore} \times \left(P(\text{worthwhile})\right)^{1.2}$$
    *(Nhân xác suất bài đáng đọc với số mũ $1.2$ giúp đẩy các bài chắc chắn hay lên top đầu và ghìm các bài không chắc chắn xuống dưới ngưỡng $T^*$)*.

### 🎯 Mục tiêu 3: Khai phá Luật kết hợp (Association Rules — Đáp ứng Rubric Báo cáo)
* **Vấn đề:** Rubric chấm bài môn học yêu cầu có phương pháp **Association Rules**.
* **Logic thuật toán:**
  * Viết script `source-code/notebooks/mine_association_rules.py`.
  * Biến đổi danh sách từ khóa trong tiêu đề và tác giả của các bài báo nhãn 4–5 thành các transactions.
  * Dùng thuật toán `frequent_patterns.apriori` hoặc `frequent_patterns.fpgrowth` (thư viện `mlxtend` hoặc script thuần python) để khai phá luật dạng: `{Keyword A, Keyword B} -> {High Priority (Label >= 4)}` với Support $\ge 0.02$, Confidence $\ge 0.70$.
  * Xuất kết quả ra `reports/association_rules.csv` để Thành viên 3 đưa vào báo cáo.

### 🎯 Mục tiêu 4: Blending 3 Mô hình (LightGBM + XGBoost + CatBoost/Ridge) & Tối ưu Ngưỡng $T^*$
* Chạy 5-Fold Stratified CV nghiêm ngặt.
* Tối ưu ngưỡng $T^*$ bằng Grid Search trên Out-of-Fold.
* Điều kiện hoàn thành: **CV Mean Score trên máy phải vượt mốc V2 (`0.61601`), hướng tới $\ge 0.625$**.
* Xuất file: `source-code/submissions/submission_v3_lambdarank.csv`.

---

## 3. Danh sách File Cần Tạo & Chỉnh Sửa

| Tác vụ | File cụ thể | Mô tả hành động |
| :--- | :--- | :--- |
| **Tạo mới** | `source-code/src/features_v3.py` | Pipeline trích xuất đặc trưng mới: SBERT + SVD + K-Means + Out-of-fold Author Stats + Bayesian Smoothed Venue. |
| **Tạo mới** | `source-code/src/train_v3.py` | Script huấn luyện V3 với Two-stage cải tiến (Stage 1 Classify, Stage 2 Ranking/Regression) + 5-Fold CV + sinh submission. |
| **Tạo mới** | `source-code/notebooks/mine_association_rules.py` | Script khai phá luật kết hợp Apriori/FP-Growth và xuất file CSV phục vụ rubric báo cáo. |
| **Cập nhật** | `reports/experiment_log.csv` | Ghi thêm dòng `experiment_id = 3` với các thông số CV, Q, R, $T^*$ tương ứng. |
| **Sinh mới** | `source-code/submissions/submission_v3_lambdarank.csv` | File nộp bài Kaggle chính thức của đợt 3. |

---

## 4. Hướng dẫn Từng Bước Cho Agent 2 (Cline) Thực Hiện

### Bước 1: Cài đặt bổ sung thư viện nhẹ (nếu cần)
```bash
source .venv/bin/activate
pip install mlxtend catboost
```
*(Nếu cài `catboost` gặp vấn đề tương thích thì fallback dùng Ensemble giữa LightGBM + XGBoost + Ridge/ElasticNet)*.

### Bước 2: Hiện thực `source-code/src/features_v3.py`
Cline cần xây dựng class `AuthorVenueEncoder` đảm bảo tính chất **Fit chỉ trên Train Split**:
```python
class AuthorVenueEncoder:
    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray):
        # Tính toán author_rating_map và venue_smoothed_map từ df_train
        ...
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Map vào df (nếu tác giả chưa từng xuất hiện, fill bằng global_mean)
        ...
```

### Bước 3: Hiện thực `source-code/src/train_v3.py`
1. Đọc dữ liệu `train.csv`, `test.csv` và 2 file numpy embedding đã có sẵn:
   - `source-code/data/train_embeddings_minilm.npy`
   - `source-code/data/test_embeddings_minilm.npy`
2. Thiết lập `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
3. Trong mỗi fold:
   - Fit feature extractor trên `train_fold`.
   - Transform cho `train_fold` và `val_fold`.
   - Huấn luyện Stage 1 (Binary Classification: $y > 1$).
   - Huấn luyện Stage 2 (Ranking/Regressor: dự đoán continuous priority).
   - Lưu OOF probabilities và OOF raw ratings.
4. Quét ngưỡng tối ưu $T^* \in [0.20, 0.70]$ bằng hàm `compute_competition_metric` trong `src/metric.py`.
5. Đánh giá: So sánh điểm OOF Score với mốc V2 cũ:
   - Điểm OOF V2: `0.61256` | Mean Folds V2: `0.61601`.
   - **Mục tiêu V3:** OOF $\ge 0.620$, Mean Folds $\ge 0.625$.
6. Fit toàn bộ mô hình trên 100% dữ liệu train, predict trên test set.
7. Áp dụng công thức ưu tiên và lọc:
   $$\text{priorities} = \text{where}(P \ge T^*, \text{Rating} \cdot P^{1.2}, 0.0)$$
8. Dùng `save_submission(...)` từ `src.utils` để lưu vào `source-code/submissions/submission_v3_lambdarank.csv`.

### Bước 4: Chạy khai phá Luật kết hợp `mine_association_rules.py`
* Chạy script để tìm ra các luật kết hợp phổ biến của các bài báo nhãn 4–5.
* Lưu file kết quả vào `reports/association_rules.csv`.

### Bước 5: Kiểm tra tính hợp lệ & Cập nhật Nhật ký
* Kiểm tra file submission:
  - Đúng 596 dòng dữ liệu (+ 1 dòng header `id,priority`).
  - Không có giá trị `NaN`, `Inf`.
  - Tỷ lệ bài được gán `priority > 0` nằm trong khoảng **55% – 70%** (khoảng 350 – 420 bài).
* Thêm kết quả vào `reports/experiment_log.csv`.
* Commit và push lên GitHub repo.

---

## 5. Quy chuẩn Đảm bảo Chất lượng (Quality Checklist)
- [ ] Không có Data Leakage: Không fit target encoder hoặc TF-IDF trước khi chia fold.
- [ ] Tính tái lập (Reproducibility): `set_seed(42)` ở đầu mọi script.
- [ ] Format file nộp chuẩn xác với Kaggle `sample_submission.csv`.
- [ ] Git commit thông điệp rõ ràng: `feat(v3): ...`
