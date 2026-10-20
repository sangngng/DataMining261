# KẾ HOẠCH CHI TIẾT & BẢNG PHÂN CÔNG NHIỆM VỤ (WBS)
## BÀI TẬP LỚN MÔN KHAI PHÁ DỮ LIỆU (CO3029 - HK241)
**Đề tài**: Xây dựng hệ thống gợi ý bài báo khoa học cá nhân hóa  
**Cuộc thi Kaggle**: [Assignment Data Mining - 261](https://www.kaggle.com/competitions/assignment-data-mining-261/overview)  
**Thời gian thực hiện**: 20/10/2026 – 14/12/2026 (8 Tuần)  
**Quy mô nhóm**: 3 Thành viên  

---

## 1. THÀNH VIÊN VÀ PHÂN VAI TRÒ CHỦ CHỐT

| Thành viên | Vai trò | Trọng tâm Kỹ thuật (Code) | Trọng tâm Báo cáo (Report) |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** *(Bạn - Sang)* | **Team Lead / ML Engineer** | - Quản trị GitHub, kiến trúc Two-Stage Pipeline.<br>- Tối ưu hóa mô hình cây (LightGBM, XGBoost, CatBoost).<br>- Tối ưu hóa ngưỡng cắt $T^*$ trên OOF.<br>- Quản lý nộp bài Kaggle. | Chịu trách nhiệm **Mục 5** (Thiết kế & cài đặt giải pháp) + Hướng dẫn tái lập kết quả (**Reproducibility**). |
| **Thành viên 2** | **Feature Engineer / NLP Specialist** | - Trích xuất đặc trưng văn bản nâng cao (TF-IDF, Sentence-Transformers, SciBERT).<br>- Phân cụm chủ đề đề tài (Topic Clustering bằng K-Means).<br>- Khai thác mạng lưới tác giả (Author rating/co-authorship) và Venue. | Chịu trách nhiệm **Mục 4** (Insights từ dữ liệu) và phần Đặc trưng trong **Mục 5**. |
| **Thành viên 3** | **Data Analyst / Report Lead** | - Thực hiện Exploratory Data Analysis (EDA) chuyên sâu.<br>- Cài đặt Baseline đối chứng và các phương pháp bổ trợ (Association Rules, Random Forest).<br>- Phân tích lỗi (Error Analysis). | Quản lý format chung (LaTeX/Word - **Mục 1**), viết **Mục 2** (Problem Formulation), **Mục 3** (Data Analysis), **Mục 6** (Evaluation & Error Analysis) và **Mục 7** (Conclusion). |

---

## 2. LỘ TRÌNH 8 TUẦN CHI TIẾT (TIMELINE & MILESTONES)

```
[Tuần 1: 20/10 - 26/10] ──> Setup, Baseline Submission & Deep EDA
[Tuần 2: 27/10 - 02/11] ──> Feature Engineering 1.0 (NLP & Metadata) + Dàn ý Báo cáo
[Tuần 3: 03/11 - 09/11] ──> Topic Clustering & Association Rules (Phương pháp môn học)
[Tuần 4: 10/11 - 16/11] ──> Nâng cấp Mô hình (XGBoost, CatBoost, LightGBM)
[Tuần 5: 17/11 - 23/11] ──> Ensemble, Stacking & Tối ưu hóa Ngưỡng cắt T*
[Tuần 6: 24/11 - 30/11] ──> Error Analysis & Chốt 2 Bài nộp Kaggle (Freeze LB)
[Tuần 7: 01/12 - 07/12] ──> Hoàn thiện toàn bộ Báo cáo & Kiểm tra Reproducibility
[Tuần 8: 08/12 - 14/12] ──> Review chéo, Nộp Báo cáo & Đóng gói Repo
```

---

### GIAI ĐOẠN 1: KHỞI ĐỘNG, BASELINE & KHÁM PHÁ DỮ LIỆU (TUẦN 1 - 2)

#### TUẦN 1 (20/10/2026 – 26/10/2026): Thiết lập nền tảng, Nộp Baseline & Hoàn thành EDA
* **Mục tiêu tuần:** Có điểm số Public Leaderboard đầu tiên, cả 3 thành viên clone repo về chạy thành công, hoàn thành notebook EDA.
* **Nhiệm vụ cụ thể:**
  * **Thành viên 1 (Leader):**
    * Nộp file `submissions/submission_two_stage.csv` lên Kaggle để lấy baseline đầu tiên.
    * Gửi link GitHub cho 2 bạn, hướng dẫn các bạn tạo virtual environment và cài đặt thư viện.
    * Thiết lập bảng theo dõi nhật ký thực nghiệm (`reports/experiment_log.csv`).
  * **Thành viên 2:**
    * Clone repo về máy, kiểm tra chạy script `python main.py` độc lập.
    * Nghiên cứu thư viện `sentence-transformers` và các pre-trained models phù hợp cho bài báo khoa học (`all-MiniLM-L6-v2`, `allenai/scibert_scivocab_uncased`).
  * **Thành viên 3:**
    * Chạy toàn diện notebook `notebooks/01_eda.ipynb`.
    * Vẽ và lưu ít nhất 6 biểu đồ chất lượng cao vào `reports/figures/`:
      1. Phân phối nhãn `Label` (tỷ lệ nhãn 1 vs nhãn 2-5).
      2. Mối quan hệ giữa Năm xuất bản (`year`) và `Label`.
      3. Top 20 hội nghị (`venue`) xuất hiện nhiều nhất và điểm trung bình tương ứng.
      4. Phân phối số lượng từ trong tiêu đề (`title word count`) theo từng nhãn.
      5. WordCloud / N-gram phân tích các từ khóa xuất hiện nhiều ở bài nhãn 4-5 so với nhãn 1.
      6. Phân tích tác giả: Tác giả nào có số lượng bài nhiều nhất và gu của nhà nghiên cứu với các tác giả đó.
* **Họp nhóm cuối tuần 1:** Đánh giá điểm Baseline trên Kaggle và chốt các phát hiện chính từ EDA.

---

#### TUẦN 2 (27/10/2026 – 02/11/2026): Trích xuất đặc trưng đợt 1 & Khởi tạo khung Báo cáo
* **Mục tiêu tuần:** Tăng điểm CV nội bộ thông qua Semantic Text Features và Metadata Features.
* **Nhiệm vụ cụ thể:**
  * **Thành viên 2 (Feature Lead):**
    * Tạo module `src/embeddings.py`: Tải pre-trained model và encode toàn bộ `title` thành vector embedding (vector 384 hoặc 768 chiều).
    * Tạo hàm lưu vector embedding ra cache dạng `.npy` để không phải tính lại nhiều lần.
    * Xây dựng đặc trưng metadata: Số lượng tác giả, kiểm tra có mã DOI hay không (`has_doi`).
  * **Thành viên 1 (Modeling Lead):**
    * Tích hợp vector embedding từ Thành viên 2 vào `src/features.py`.
    * Chạy 5-Fold Cross Validation đối chiếu giữa TF-IDF cũ và Sentence-BERT mới.
    * Xuất file submission đợt 2 nếu CV tăng và nộp lên Kaggle.
  * **Thành viên 3 (Report Lead):**
    * Khởi tạo template Báo cáo (file Word hoặc LaTeX theo chuẩn của khoa).
    * Soạn thảo thô **Mục 2: Problem Formulation & Domain Understanding** (định nghĩa nhà nghiên cứu cần gì, phân tích tại sao metric dùng Harmonic Mean của Q và R).
    * Soạn thảo thô **Mục 3: Data Analysis** (chèn các biểu đồ từ Tuần 1 và giải thích số liệu).
* **Họp nhóm cuối tuần 2:** Thống nhất các đặc trưng mang lại hiệu quả cao trên CV.

---

### GIAI ĐOẠN 2: PHƯƠNG PHÁP CỐT LÕI MÔN HỌC & MÔ HÌNH HÓA CHUYÊN SÂU (TUẦN 3 - 5)

#### TUẦN 3 (03/11/2026 – 09/11/2026): Topic Clustering & Association Rules
* **Mục tiêu tuần:** Đáp ứng xuất sắc yêu cầu bắt buộc của môn học (Áp dụng Phân cụm và Luật kết hợp).
* **Nhiệm vụ cụ thể:**
  * **Thành viên 2:**
    * **Clustering:** Nâng cấp thuật toán K-Means và Hierarchical Clustering trên text embedding để gom cụm các bài báo thành $K$ chủ đề (ví dụ $K=8, 12, 16$).
    * Tính khoảng cách từ mỗi bài báo tới tâm các cụm làm đặc trưng đầu vào cho mô hình phân loại.
    * Gán tên cho từng cụm (ví dụ: Cụm Reinforcement Learning, Cụm Formal Verification, Cụm Natural Language Processing,...).
  * **Thành viên 3:**
    * **Association Rules:** Sử dụng thuật toán Apriori hoặc FP-Growth để tìm các tập phổ biến (Frequent Itemsets) giữa các tác giả và các từ khóa nghiên cứu thường đi cùng nhau trong các bài nhãn 4-5.
    * Viết phần giải thích lý thuyết và trực quan hóa các luật kết hợp tìm được.
  * **Thành viên 1:**
    * Nhúng các đặc trưng cụm và luật kết hợp vào pipeline huấn luyện.
    * Đo lường mức độ đóng góp của từng nhóm đặc trưng thông qua Feature Importance của LightGBM.
    * Cập nhật file nộp lên Kaggle.

---

#### TUẦN 4 (10/11/2026 – 16/11/2026): Mở rộng Mô hình (XGBoost, CatBoost & Learning to Rank)
* **Mục tiêu tuần:** Đa dạng hóa thuật toán để chuẩn bị cho bước Ensemble, bứt phá bảng xếp hạng.
* **Nhiệm vụ cụ thể:**
  * **Thành viên 1:**
    * Cài đặt thêm **XGBoost** và **CatBoost** cho cả 2 giai đoạn (Classification & Regression).
    * Thử nghiệm mô hình chuyên biệt cho xếp hạng: **LightGBM LambdaMART** (Objective: `lambdarank`).
    * So sánh hiệu năng giữa hồi quy thông thường (MSE) và LambdaMART trên metric của cuộc thi.
  * **Thành viên 2:**
    * Tinh chỉnh đặc trưng tác giả: Tính toán Out-of-Fold Target Encoding cho `authors` và `venue` (ví dụ: điểm trung bình lịch sử của tác giả, điểm lớn nhất của các đồng tác giả). Lưu ý kỹ thuật để tránh data leakage.
  * **Thành viên 3:**
    * Xây dựng bảng đối chứng các mô hình cơ sở (**Baseline Comparison Table**):
      * Baseline 1: Dummy Recommender (gán nhãn trung bình hoặc ngẫu nhiên).
      * Baseline 2: Logistic Regression + TF-IDF.
      * Baseline 3: Random Forest.
      * Proposed Method: Two-Stage GBDT Recommender.
    * Bắt đầu viết nội dung cho **Mục 4: Insights** dựa trên kết quả thực nghiệm.

---

#### TUẦN 5 (17/11/2026 – 23/11/2026): Tối ưu hóa Siêu tham số & Ensemble / Stacking
* **Mục tiêu tuần:** Đạt điểm số Local CV và Kaggle Leaderboard cao nhất của nhóm.
* **Nhiệm vụ cụ thể:**
  * **Thành viên 1:**
    * Sử dụng thư viện `optuna` để tự động dò tìm bộ siêu tham số tối ưu (Hyperparameter Tuning) cho mô hình tốt nhất (`num_leaves`, `learning_rate`, `subsample`, `feature_fraction`, `reg_alpha`, `reg_lambda`).
    * Xây dựng cơ chế **Ensemble (Blending / Weighted Average)** kết hợp dự đoán giữa LightGBM, CatBoost và XGBoost.
    * Chạy thuật toán tìm kiếm lưới (Grid Search) tối ưu hóa ngưỡng cắt $T^* \in [0.1, 0.9]$ trên toàn bộ Out-Of-Fold predictions.
  * **Thành viên 2:**
    * Phân tích độ quan trọng của đặc trưng (**Feature Importance Analysis**): Sử dụng SHAP values hoặc LightGBM feature gain để xác định 15 đặc trưng quyết định nhất tới điểm số.
    * Xuất biểu đồ SHAP summary plot cho báo cáo.
  * **Thành viên 3:**
    * Hỗ trợ chạy các seed ngẫu nhiên khác nhau (42, 123, 999) để kiểm tra độ ổn định của mô hình (Fair evaluation & variance analysis).

---

### GIAI ĐOẠN 3: PHÂN TÍCH LỖI, CHỐT BÀI NỘP & HOÀN THIỆN BÁO CÁO (TUẦN 6 - 8)

#### TUẦN 6 (24/11/2026 – 30/11/2026): Phân tích Lỗi (Error Analysis) & Đóng băng mô hình
* **Mục tiêu tuần:** Hoàn thành toàn bộ thực nghiệm, đóng băng code mô hình, chọn 2 bài nộp chính thức trên Kaggle.
* **Nhiệm vụ cụ thể:**
  * **Cả nhóm:** 
    * Kiểm tra bảng xếp hạng Kaggle, vào mục **Submissions** và chủ động tick chọn **2 bài nộp chính thức (Select 2 Final Submissions)**:
      1. Bài 1: Mô hình có điểm Public Leaderboard cao nhất.
      2. Bài 2: Mô hình có điểm 5-Fold Cross Validation nội bộ cao và ổn định nhất.
  * **Thành viên 3 (Lead Error Analysis):**
    * So sánh nhãn thật $y$ và nhãn dự đoán trên tập Out-of-Fold.
    * Lọc ra và phân tích chi tiết:
      * **Trường hợp False Positive nghiêm trọng:** Những bài nhãn 1 mà mô hình lại cho điểm ưu tiên cao (tìm hiểu vì sao: do tiêu đề chứa từ khóa gây nhiễu hay do tác giả nổi tiếng?).
      * **Trường hợp False Negative nghiêm trọng:** Những bài nhãn 5 mà mô hình bị lọc nhầm thành priority 0 (lý giải nguyên nhân).
    * Đưa các trường hợp cụ thể này vào bảng phân tích lỗi cho báo cáo (tiêu chí này chiếm tới **2.0 điểm** trong rubric).
  * **Thành viên 1 & 2:**
    * Dọn dẹp code, refactor các module trong `src/`, xóa các file tạm và comment code rõ ràng.

---

#### TUẦN 7 (01/12/2026 – 07/12/2026): Viết Báo cáo Tổng lực & Kiểm tra Tái lập (Reproducibility)
* **Mục tiêu tuần:** Hoàn thành 100% bản thảo báo cáo, kiểm tra việc tái lập mã nguồn từ con số 0.
* **Nhiệm vụ cụ thể:**
  * **Thành viên 3:**
    * Ghép tất cả các mục của báo cáo lại thành một file hoàn chỉnh.
    * Soát lỗi chính tả, thuật ngữ khoa học, đánh số thứ tự hình vẽ, bảng biểu và danh mục tài liệu tham khảo (References chuẩn APA/IEEE).
    * Điền đầy đủ: Tên nhóm, Danh sách thành viên, MSSV, Bảng phân công đóng góp cá nhân.
  * **Thành viên 1:**
    * Viết hoàn chỉnh `README.md` trên GitHub: mô tả cách clone, tạo môi trường, lệnh chạy sinh ra đúng file submission được chọn.
    * Đóng dấu Git Tag: `git tag -a v1.0-final -m "Final submission release"` và push lên GitHub.
    * Lấy Commit Hash cuối cùng để điền vào trang bìa báo cáo.
  * **Thành viên 2:**
    * Thực hiện bài test tái lập độc lập: Đóng vai trò là "người lạ", clone repo về một máy tính khác (hoặc tạo folder trống mới), chạy lại từ đầu bằng lệnh trong `README.md` xem có sinh ra đúng file submission không.

---

#### TUẦN 8 (08/12/2026 – 14/12/2026): Rà soát cuối cùng & Nộp bài
* **Mục tiêu tuần:** Xuất bản PDF báo cáo, kiểm tra quyền truy cập repo của Thầy, hoàn tất nộp bài trên LMS/Kaggle.
* **Nhiệm vụ cụ thể:**
  * **Cả nhóm:**
    * Đọc chéo toàn bộ bài báo cáo từ đầu đến cuối một lần nữa.
    * Đảm bảo repo GitHub đã được cấp quyền cho Thầy (`tuvn@hcmut.edu.vn`) nếu là repo Private.
    * Xuất file báo cáo sang định dạng `.pdf`.
    * Nộp bài lên cổng LMS của trường theo đúng hướng dẫn của Thầy và đúng hạn.

---

## 3. CHECKLIST ĐÁNH GIÁ THEO RUBRIC CHẤM ĐIỂM (10 ĐIỂM BÁO CÁO)

| Tiêu chí | Điểm tối đa | Trách nhiệm chính | Nội dung cần kiểm tra trước khi nộp |
| :--- | :---: | :---: | :--- |
| **1. Presentation & Clarity** | **1.0đ** | Thành viên 3 | Font chữ đồng nhất, hình ảnh sắc nét, bảng biểu có chú thích, trích dẫn tài liệu chuẩn. |
| **2. Problem Formulation** | **1.5đ** | Thành viên 3 | Giải thích sâu về người dùng (nhà nghiên cứu), ý nghĩa của Relevance $Q$, Coverage $R$ và Harmonic Mean. |
| **3. Data Analysis (EDA)** | **1.5đ** | Thành viên 3 | Đầy đủ biểu đồ phân phối nhãn, phân tích text, tác giả, năm, giải thích các quyết định tiền xử lý. |
| **4. Insights from Data** | **1.5đ** | Thành viên 2 | Đúc kết các phát hiện thực tế từ dữ liệu, phân biệt rõ quan sát thực tế với giả định chủ quan. |
| **5. Solution Design & Implementation** | **2.0đ** | Thành viên 1 | Trình bày kiến trúc Two-Stage, áp dụng ít nhất 1 phương pháp môn học (Clustering/Classification/Regression), code tái lập được trên GitHub. |
| **6. Evaluation, Error Analysis & Fairness** | **2.0đ** | Thành viên 3 & 1 | Thiết kế 5-Fold CV chặt chẽ, so sánh với Baseline, phân tích chi tiết các ca dự đoán lỗi, chứng minh không overfit Public LB. |
| **7. Conclusions & General Insights** | **0.5đ** | Thành viên 3 | Đúc kết đóng góp, nêu rõ các hạn chế (limitations) và hướng phát triển tiếp theo. |

---

## 4. QUY TẮC PHỐI HỢP NHÓM (TEAM WORKING RULES)

1. **Lịch họp cố định:** Họp online/offline định kỳ **tối Chủ nhật hàng tuần (20h00 - 21h00)** để rà soát tiến độ tuần cũ và giao việc tuần mới.
2. **Quy tắc Git:**
   * Không bao giờ commit trực tiếp lên nhánh `main`.
   * Tạo nhánh theo cú pháp: `feature/<tên_tính_năng>` hoặc `eda/<tên_task>`.
   * Chỉ merge vào `main` sau khi đã chạy kiểm tra không gây lỗi code chung.
3. **Quy tắc nộp Kaggle:**
   * Mỗi ngày tối đa 1–2 lần nộp, chỉ nộp khi điểm Local CV thực sự cải thiện rõ rệt.
   * Ghi lại đầy đủ thông tin vào nhật ký nộp bài: Ngày, Tên file, Điểm CV, Điểm Public LB, Mô tả thay đổi.
