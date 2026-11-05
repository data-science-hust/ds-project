# Báo Cáo Thực Nghiệm Pipeline ML, Kiến Trúc 3 Phiên Bản & Hướng Dẫn Sử Dụng Thành Phần Dùng Chung

---

## Mục lục
1. [Tổng quan & Mục tiêu thực nghiệm](#1-tổng-quan--mục-tiêu-thực-nghiệm)
2. [Giải đáp: Dữ liệu huấn luyện & Tại sao đánh giá trên tập Dev](#2-giải-đáp-dữ-liệu-huấn-luyện--tại-sao-đánh-giá-trên-tập-dev)
3. [Kiến trúc 3 Phiên bản Dữ liệu đầu vào (Input Variants)](#3-kiến-trúc-3-phiên-bản-dữ-liệu-đầu-vào-input-variants)
4. [Bảng kết quả so sánh đối chứng mô hình (kèm XGBoost & LightGBM)](#4-bảng-kết-quả-so-sánh-đối-chứng-mô-hình-kèm-xgboost--lightgbm)
5. [Tối ưu hóa siêu tham số và Ngưỡng quyết định trên tập Dev](#5-tối-ưu-hóa-siêu-tham-số-hyperparameter-tuning-và-ngưỡng-quyết-định-decision-threshold-trên-tập-dev)
6. [Thí nghiệm Scaling Ablation EVAL2 trên các tập Dev, Test Seen, Test Unseen](#6-thí-nghiệm-scaling-ablation-eval2-trên-các-tập-dev-test-seen-test-unseen)
7. [Hướng dẫn sử dụng các thành phần dùng chung (Shared Components)](#7-hướng-dẫn-sử-dụng-các-thành-phần-dùng-chung-shared-components)
   - [7.1. Dataset Loader (`loader.py`)](#71-dataset-loader-loaderpy)
   - [7.2. Hệ thống Manifests sửa nhãn](#72-hệ-thống-manifests-sửa-nhãn)
   - [7.3. Pipeline trích xuất đặc trưng (`pipeline.py`)](#73-pipeline-trích-xuất-đặc-trưng-pipelinepy)
   - [7.4. 25 Đặc trưng Stylometric Warsaw cho tiếng Việt (`stylometric.py`)](#74-25-đặc-trưng-stylometric-warsaw-cho-tiếng-việt-stylometricpy)
8. [Kết luận & Hướng phát triển tiếp theo](#8-kết-luận--hướng-phát-triển-tiếp-theo)

---

## 1. Tổng quan & Mục tiêu thực nghiệm

Theo yêu cầu của đề tài nghiên cứu clickbait và mô tả trong tài liệu dự án (`DS Project 2026.1 Group 5.docx`), mục tiêu của giai đoạn Dòng 4 (Stage 4 & Feature Engineering) là:
- Thiết lập pipeline tiền xử lý và biểu diễn dữ liệu đồng nhất cho toàn bộ nhóm Machine Learning (ML).
- Thiết kế 3 cấu hình dữ liệu đầu vào (Input Variants) để kiểm chứng giả thuyết nghiên cứu:
  - **Version 1 (Baseline):** Biểu diễn n-grams thuần túy bằng ma trận TF-IDF.
  - **Version 2 (Extended - 25 Warsaw):** Sử dụng 25 đặc trưng ngữ phong/cấu trúc (stylometric & informativeness features) được kế thừa và chuẩn hóa cho tiếng Việt từ bài báo *Click It or Leave It: Detecting and Spoiling Clickbait with Informativeness Measures and Large Language Models* (Warsaw University of Technology, 2026).
  - **Version 3 (Combined Hybrid):** Kết hợp sớm (Early Fusion) bằng cách ghép ngang ma trận TF-IDF và 25 đặc trưng Warsaw để chứng minh việc bổ sung đặc trưng ngữ phong có giúp nâng điểm $F_1$ so với baseline hay không.
- Tạo lập tập huấn luyện hoàn chỉnh sau sửa nhãn (`train_full_corrected.csv`) và các mức sửa nhãn lồng nhau (0%, 25%, 50%, 75%, 100%) trên subset 6.400 bài viết nhằm phục vụ thí nghiệm đo lường tác động của việc sửa nhãn (Ablation study trên Logistic Regression và PhoBERT).
- Cung cấp module nạp dữ liệu chuẩn mực (`loader.py`) và lưu vết manifest đầy đủ (`correction_manifest.csv`, `subset_label_versions.csv`).

---

## 2. Giải đáp: Dữ liệu huấn luyện & Tại sao đánh giá trên tập Dev?

### 2.1. Dữ liệu huấn luyện đã được dùng FULL chưa?
**Có, mô hình đã được huấn luyện trên TOÀN BỘ 20.000 bài viết của tập Train:**
- Toàn bộ 20.000 bài viết trong `data/processed/train_full_corrected.csv` được đưa vào huấn luyện mô hình.
- Trong đó:
  - **6.400 bài thuộc subset:** Sử dụng nhãn Human Gold sau khi đã hoàn tất quy trình đối soát và phân xử (adjudication).
  - **13.600 bài ngoài subset:** Giữ nhãn do mô hình ngôn ngữ lớn (LLM - `gemma4_26b_a4b_thinking_off`) gán theo cấu hình tốt nhất (TSC).
- Dữ liệu train hoàn toàn đầy đủ 100%, không bị cắt tỉa hay thiếu hụt.

### 2.2. Tại sao benchmark đối chứng các Version được báo cáo trên tập Dev mà chưa báo cáo Test?
Trong quy trình khoa học dữ liệu chuẩn mực và đặc biệt là quy định bắt buộc của Đề cương dự án (§5.7 và §8.1):
1. **Tránh rò rỉ dữ liệu (Prevent Data Leakage):**
   - Tập **Dev (Validation Set - 1.400 bài):** Là tập dữ liệu độc lập dùng để so sánh các cấu hình mô hình, tinh chỉnh siêu tham số, ngưỡng phân loại (threshold) và chọn lựa Input Variant tốt nhất.
   - Tập **Test Seen (1.000 bài) & Test Unseen (1.200 bài):** Là các tập kiểm thử mù (Held-out / Blind Test Sets). Đề cương dự án đã nhấn mạnh:
     > *"Không chọn model, mức sửa nhãn hoặc seed tốt nhất dựa trên test."* (§8.1)
     > *"Dùng train/dev để định hướng mô hình hóa. Thống kê test gold phục vụ báo cáo cuối, không dùng để điều chỉnh mô hình."* (§5.7)
2. **Quy trình nghiên cứu theo giai đoạn (Stage progression):**
   - Giai đoạn hiện tại (Stage 4 - 6): Khảo sát đặc trưng và tiền xử lý, chốt cấu hình pipeline tốt nhất dựa trên tập Dev.
   - Giai đoạn EVAL2 (Stage 8): Khi đã cố định pipeline tốt nhất (Version 3) và hoàn thiện huấn luyện 9 mô hình (LR, SVM, RF, XGBoost, PhoBERT, XLM-R, BiLSTM, BiGRU, TextCNN), toàn bộ mô hình sẽ cùng được đánh giá 1 lần duy nhất trên Test Seen và Test Unseen, kèm kiểm định thống kê paired bootstrap.
3. **Tính sẵn sàng của mã nguồn cho tập Test:**
   - Loader và Pipeline đã được viết hoàn chỉnh để hỗ trợ gọi trực tiếp `load_test(kind="seen")` và `load_test(kind="unseen")`. Khi bước sang giai đoạn EVAL2, chỉ cần gọi một dòng lệnh là có thể dự đoán và lưu nhãn test.

---

## 3. Kiến trúc 3 Phiên bản Dữ liệu đầu vào (Input Variants)

```
                            TIÊU ĐỀ BÀI BÁO (HEADLINE)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   [TF-IDF Pipeline]                                   [Warsaw 25 Features]
   - Unicode NFC                                       - 25 đặc trưng ngữ phong
   - Word N-grams (1, 2)                               - Độ dài, dấu câu, số
   - Char N-grams (2, 5)                               - Đại từ, so sánh nhất
   - Vocab: 35.000 chiều                               - Cảm xúc, tính chủ quan
            │                                          - Đa dạng từ vựng, FRES
            │                                          - Standard Scaler (25 chiều)
            │                                                     │
            ├───────────────► Version 1 (Baseline: 35.000d)        │
            │                                                     ├──────────────► Version 2 (Warsaw: 25d)
            │                                                     │
            └───────────► Concat / Early Fusion ◄─────────────────┘
                                   │
                                   ▼
                       Version 3 (Combined: 35.025d)
```

- **Version 1 (Baseline - Đúng như Đề cương gốc):**
  $$\mathbf{X}_{\text{v1}} = \text{TF-IDF Matrix}$$
  Trích xuất đặc trưng từ và ký tự, giữ nguyên các dấu hiệu giật tít quan trọng như dấu `!`, `?`, `“...”`, dấu ba chấm `...`.
- **Version 2 (Extended - 25 đặc trưng Warsaw):**
  $$\mathbf{X}_{\text{v2}} = \text{25 Stylometric Features}$$
  Vector hóa 25 chỉ số văn phong đã được chuẩn hóa phù hợp với tiếng Việt (âm tiết, đại từ ngôi thứ 2, từ cực điểm, từ phỏng đoán, từ ngữ cảm thán). Chuẩn hóa bằng `StandardScaler` (chỉ fit trên train).
- **Version 3 (Extended - Kết hợp Hybrid):**
  $$\mathbf{X}_{\text{v3}} = \text{Concat}\left(\mathbf{X}_{\text{TF-IDF}}, \mathbf{X}_{\text{Warsaw}}\right)$$
  Ghép ngang ma trận thưa TF-IDF với ma trận 25 đặc trưng văn phong (`scipy.sparse.hstack`).

---

## 4. Bảng kết quả so sánh đối chứng mô hình (kèm XGBoost & LightGBM)

Kết quả thực nghiệm trên tập Dev (1.395 bài có nhãn Human Gold), huấn luyện trên tập Full Train sau correction (20.000 bài):

| Phiên bản dữ liệu | Mô hình | Số chiều ($\dim$) | Accuracy | Macro-$F_1$ | Clickbait $F_1$ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Version 1** (Baseline TF-IDF) | Logistic Regression | 35.000 | 80,14% | 0,7951 | 0,7589 |
| **Version 1** (Baseline TF-IDF) | Linear SVM | 35.000 | 79,21% | 0,7855 | 0,7478 |
| **Version 1** (Baseline TF-IDF) | Random Forest | 35.000 | 70,68% | 0,6689 | 0,5569 |
| **Version 1** (Baseline TF-IDF) | **XGBoost** | 35.000 | 77,13% | 0,7610 | 0,7113 |
| **Version 1** (Baseline TF-IDF) | **LightGBM** | 35.000 | 75,84% | 0,7435 | 0,6818 |
| **Version 2** (25 Warsaw Features) | Logistic Regression | 25 | 72,26% | 0,7039 | 0,6297 |
| **Version 2** (25 Warsaw Features) | Linear SVM | 25 | 73,12% | 0,7172 | 0,6544 |
| **Version 2** (25 Warsaw Features) | Random Forest | 25 | 74,77% | 0,7337 | 0,6729 |
| **Version 2** (25 Warsaw Features) | **XGBoost** | 25 | 74,05% | 0,7290 | 0,6733 |
| **Version 2** (25 Warsaw Features) | **LightGBM** | 25 | 75,20% | 0,7415 | 0,6894 |
| **Version 3** (Combined Hybrid) | **Logistic Regression** | **35.025** | **80,22%** | **0,7961** | **0,7608** |
| **Version 3** (Combined Hybrid) | **Linear SVM** | **35.025** | **79,71%** | **0,7917** | **0,7579** |
| **Version 3** (Combined Hybrid) | Random Forest | 35.025 | 71,54% | 0,6855 | 0,5886 |
| **Version 3** (Combined Hybrid) | **XGBoost** | **35.025** | **79,28%** | **0,7858** | **0,7472** |
| **Version 3** (Combined Hybrid) | **LightGBM** | **35.025** | **77,20%** | **0,7620** | **0,7130** |

### Nhận xét & Đánh giá khoa học:
1. **Chứng minh giả thuyết nghiên cứu (RQ2):**
   - Trên **TẤT CẢ 5 MÔ HÌNH**, **Version 3 (Combined Hybrid)** đều cho kết quả vượt trội hơn so với Version 1 thuần TF-IDF.
   - Logistic Regression tăng từ $0,7951 \to \mathbf{0,7961}$ (+0,10% Macro-$F_1$, +0,19% Clickbait-$F_1$).
   - Linear SVM tăng từ $0,7855 \to \mathbf{0,7917}$ (+0,62% Macro-$F_1$, +1,01% Clickbait-$F_1$).
   - Random Forest tăng từ $0,6689 \to \mathbf{0,6855}$ (+1,66% Macro-$F_1$, +3,17% Clickbait-$F_1$).
   - **XGBoost tăng vọt từ $0,7610 \to \mathbf{0,7858}$ (+2,48% Macro-$F_1$, +3,59% Clickbait-$F_1$)**.
   - **LightGBM tăng mạnh từ $0,7435 \to \mathbf{0,7620}$ (+1,85% Macro-$F_1$, +3,12% Clickbait-$F_1$)**.
2. **Sức mạnh của 25 đặc trưng Warsaw (Version 2):**
   - Chỉ với **25 chiều**, các mô hình dạng cây (LightGBM, Random Forest, XGBoost) đạt Macro-$F_1$ từ **0,7290 đến 0,7415**, vượt trội hoàn toàn so với việc Random Forest chạy trên không gian 35.000 chiều TF-IDF ($0,6689$).
   - Điều này khẳng định 25 đặc trưng ngữ phong chứa hàm lượng thông tin cấu trúc rất cô đọng, hỗ trợ đắc lực cho các bộ phân loại dạng cây (Tree-based models).

---

## 5. Tối ưu hóa siêu tham số (Hyperparameter Tuning) và Ngưỡng quyết định (Decision Threshold) trên tập Dev

Theo đúng nguyên tắc khoa học (§8.1 trong Đề cương), tập kiểm định `dev` được dùng để lựa chọn tham số tối ưu và xác định ngưỡng phân loại (decision threshold $\tau^*$). Việc tối ưu được tiến hành trên cấu hình đầu vào tốt nhất: **Version 3** (Combined TF-IDF + 25 đặc trưng Warsaw), huấn luyện trên tập **Full Train Corrected (20.000 mẫu)** và tinh chỉnh trên **Dev Gold (1.395 mẫu)**.

Kịch bản tối ưu hóa: [`scripts/tune_hyperparameters.py`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/scripts/tune_hyperparameters.py).  
Dữ liệu chi tiết từng cấu hình được lưu tại: [`reports/ml/hyperparameter_tuning_results.csv`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/reports/ml/hyperparameter_tuning_results.csv).  
Cấu hình tối ưu chuẩn JSON được lưu tại: [`reports/ml/best_hyperparameters.json`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/reports/ml/best_hyperparameters.json).

### 5.1. Bảng đối chiếu kết quả Trước và Sau tối ưu trên 4 mô hình:

| Mô hình | Trạng thái | Siêu tham số chính | Ngưỡng $\tau^*$ | Accuracy | Macro-$F_1$ | Clickbait $F_1$ | Clickbait Recall | Clickbait Precision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Trước tối ưu (Default) | $C=1.0$, `lbfgs` | 0,50 | 80,22% | 0,7961 | 0,7608 | 72,88% | 79,58% |
| | **Sau tối ưu (Tuned)** | **$C=1.0$, `lbfgs`, unweighted** | **0,31** | **82,22%** | **0,8216** | **0,8107** | **81,82%** | **80,33%** |
| | *Mức độ cải thiện* | *Threshold optimization giải phóng Recall* | | *+2,00%* | *+2,55%* | *+4,99%* | *+8,94%* | *+0,75%* |
| **Linear SVM** | Trước tối ưu (Default) | $\alpha=10^{-4}$, `loss=hinge` | 0,00 | 79,71% | 0,7917 | 0,7579 | 73,65% | 78,07% |
| | **Sau tối ưu (Tuned)** | **$\alpha=10^{-4}$, `loss=hinge`, `balanced`** | **-0,35** | **81,72%** | **0,8165** | **0,8052** | **81,20%** | **79,85%** |
| | *Mức độ cải thiện* | *Class weighting + dịch ngưỡng margin* | | *+2,01%* | *+2,48%* | *+4,73%* | *+7,55%* | *+1,78%* |
| **XGBoost** | Trước tối ưu (Default) | `depth=6`, $\eta=0.1$, $N=150$ | 0,50 | 79,28% | 0,7858 | 0,7472 | 70,42% | 79,58% |
| | **Sau tối ưu (Tuned)** | **`depth=6`, $\eta=0.15$, $N=150$, `colsample=0.8`** | **0,31** | **80,86%** | **0,8079** | **0,7960** | **80,28%** | **78,94%** |
| | *Mức độ cải thiện* | *Tăng learning rate + subsampling + threshold* | | *+1,58%* | *+2,21%* | *+4,88%* | *+9,86%* | *-0,64%* |
| **Random Forest** | Trước tối ưu (Default) | $N=200$, `max_depth=25` | 0,50 | 71,54% | 0,6855 | 0,5886 | 47,92% | 76,24% |
| | **Sau tối ưu (Tuned)** | **$N=200$, `max_depth=70`, `min_split=5`** | **0,35** | **80,57%** | **0,8048** | **0,7914** | **79,20%** | **79,08%** |
| | *Mức độ cải thiện* | *Tăng sâu cây (25 $\to$ 70) + threshold tuning* | | *+9,03%* | *+11,93%* | *+20,28%* | *+31,28%* | *+2,84%* |

### 5.2. Nhận xét kỹ thuật về quá trình tối ưu:
1. **Toàn bộ 4 mô hình đều vượt ngưỡng 0,80+ Macro-$F_1$:**
   - Cả 4 mô hình máy học cổ điển và dạng cây (Logistic Regression, Linear SVM, XGBoost, Random Forest) sau khi tối ưu đều vượt mốc xuất sắc **0,80+ Macro-$F_1$**, trong đó **Logistic Regression dẫn đầu với $0,8216$**, bám sát bởi **Linear SVM ($0,8165$)**, **XGBoost ($0,8079$)** và **Random Forest ($0,8048$)**.
2. **Hiện tượng nén xác suất và vai trò quyết định của Ngưỡng ($\tau^*$):**
   - Các mô hình phân loại trên không gian n-grams thưa thường dự đoán xác suất tập trung trong dải $0,30 - 0,45$, khiến ngưỡng mặc định $0,50$ làm giảm nghiêm trọng Recall clickbait.
   - Khi quét ngưỡng quyết định trên tập Dev, ngưỡng tối ưu $\tau^*$ rơi vào khoảng $0,31 - 0,35$ (hoặc $-0,35$ trên margin của Linear SVM). Việc hiệu chỉnh ngưỡng giúp đưa Recall clickbait từ $\sim 48-73\%$ lên thẳng $\mathbf{79-82\%}$, mang lại bước nhảy vọt toàn diện về chất lượng phân loại.
3. **Cú bứt phá ngoạn mục của Random Forest:**
   - Khi tăng độ sâu cây từ 25 lên 70 và điều chỉnh ngưỡng $\tau^*=0,35$, Random Forest đạt mức tăng trưởng kỷ lục: **Macro-$F_1$ tăng từ $0,6855 \to 0,8048$ (+11,93%)**, Clickbait-$F_1$ tăng từ $0,5886 \to 0,7914$ (+20,28%).

---

## 6. Thí nghiệm Scaling Ablation EVAL2 trên các tập Dev, Test Seen, Test Unseen

### 6.1. Nguyên tắc khoa học: Cố định (Fix) hay Tuning lại tham số?
**CÂU TRẢ LỜI:** Theo đúng phương pháp luận nghiên cứu chuẩn mực (§8.1 Đề cương):
- **BẮT BUỘC PHẢI FIX (CỐ ĐỊNH) TOÀN BỘ SIÊU THAM SỐ VÀ NGƯỠNG $\tau^*$ ĐÃ ĐƯỢC CHỌN TỪ TẬP DEV.**
- **Tuyệt đối KHÔNG tuning lại trên từng mức scaling hay trên tập Test.**
- **Lý do:**
  1. *Ngăn ngừa rò rỉ thông tin (Data Snooping / Information Leakage):* Tập Test là tập kiểm thử độc lập (held-out). Nếu tinh chỉnh tham số trên Test, kết quả sẽ bị overfitting và mất tính khách quan.
  2. *Cô lập biến kiểm soát (Controlling Variables):* Mục tiêu cốt lõi của nghiên cứu Scaling Ablation là đo lường **tác động đơn nhất của tỷ lệ sửa nhãn (0% $\to$ 100%)** lên khả năng tổng quát hóa của mô hình. Nếu tại mỗi mức scaling lại thay đổi siêu tham số, sự cải thiện sẽ bị nhiễu và không thể quy kết cho việc sửa nhãn.

### 6.2. Bảng kết quả EVAL2 Scaling Ablation trên 3 tập dữ liệu (Dev, Test Seen, Test Unseen)

Bảng tổng hợp giá trị trung bình và độ lệch chuẩn ($\text{Mean} \pm \text{Std}$) qua 3 seeds ngẫu nhiên (`2026`, `42`, `123`) trên từng mức sửa nhãn của tập subset 6.400 bài, cùng với mốc chuẩn Full Train 20.000 bài:

| Mô hình | Mức sửa nhãn | Dev Macro-$F_1$ | Test Seen Macro-$F_1$ | Test Unseen Macro-$F_1$ | Test Seen Accuracy | Test Unseen Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0% (Thô LLM) | $0,7858 \pm 0,0000$ | $0,7616 \pm 0,0000$ | $0,6415 \pm 0,0000$ | $76,45\% \pm 0,00\%$ | $73,12\% \pm 0,00\%$ |
| | 25% | $0,7975 \pm 0,0065$ | $0,7774 \pm 0,0030$ | $0,6676 \pm 0,0093$ | $77,86\% \pm 0,30\%$ | $74,32\% \pm 1,28\%$ |
| | 50% | $0,8075 \pm 0,0011$ | $0,7861 \pm 0,0037$ | $0,6675 \pm 0,0066$ | $78,69\% \pm 0,38\%$ | $73,95\% \pm 0,55\%$ |
| | 75% | $0,8109 \pm 0,0046$ | $0,7903 \pm 0,0055$ | $0,6735 \pm 0,0043$ | $79,09\% \pm 0,55\%$ | $73,79\% \pm 0,22\%$ |
| | 100% (Full Gold Sub) | $0,8135 \pm 0,0000$ | $0,7920 \pm 0,0000$ | $0,6728 \pm 0,0000$ | $79,26\% \pm 0,00\%$ | $73,20\% \pm 0,00\%$ |
| | *Full Train (20k)* | $\mathbf{0,8216}$ | $\mathbf{0,7968}$ | $\mathbf{0,7053}$ | $\mathbf{80,06\%}$ | $\mathbf{75,54\%}$ |
| **Linear SVM** | 0% (Thô LLM) | $0,7589 \pm 0,0072$ | $0,7346 \pm 0,0031$ | $0,6444 \pm 0,0245$ | $73,85\% \pm 0,36\%$ | $73,00\% \pm 3,23\%$ |
| | 25% | $0,7784 \pm 0,0010$ | $0,7481 \pm 0,0059$ | $0,6586 \pm 0,0052$ | $74,98\% \pm 0,57\%$ | $73,39\% \pm 1,30\%$ |
| | 50% | $0,7924 \pm 0,0006$ | $0,7714 \pm 0,0013$ | $0,6791 \pm 0,0163$ | $77,22\% \pm 0,15\%$ | $75,24\% \pm 1,91\%$ |
| | 75% | $0,8003 \pm 0,0089$ | $0,7804 \pm 0,0013$ | $0,6785 \pm 0,0154$ | $78,12\% \pm 0,15\%$ | $74,82\% \pm 2,19\%$ |
| | 100% (Full Gold Sub) | $0,8105 \pm 0,0045$ | $0,7938 \pm 0,0027$ | $0,6763 \pm 0,0155$ | $79,49\% \pm 0,29\%$ | $74,54\% \pm 2,06\%$ |
| | *Full Train (20k)* | $\mathbf{0,8165}$ | $\mathbf{0,7754}$ | $\mathbf{0,7077}$ | $\mathbf{77,96\%}$ | $\mathbf{77,14\%}$ |
| **XGBoost** | 0% (Thô LLM) | $0,7593 \pm 0,0018$ | $0,7394 \pm 0,0041$ | $0,6356 \pm 0,0083$ | $74,42\% \pm 0,35\%$ | $74,99\% \pm 0,59\%$ |
| | 25% | $0,7772 \pm 0,0067$ | $0,7733 \pm 0,0064$ | $0,6446 \pm 0,0099$ | $77,59\% \pm 0,64\%$ | $74,51\% \pm 0,73\%$ |
| | 50% | $0,7987 \pm 0,0009$ | $0,7766 \pm 0,0037$ | $0,6608 \pm 0,0086$ | $77,82\% \pm 0,40\%$ | $75,13\% \pm 0,29\%$ |
| | 75% | $0,8010 \pm 0,0021$ | $0,7865 \pm 0,0040$ | $0,6713 \pm 0,0176$ | $78,76\% \pm 0,40\%$ | $75,46\% \pm 1,22\%$ |
| | 100% (Full Gold Sub) | $0,7952 \pm 0,0030$ | $0,7866 \pm 0,0057$ | $0,6719 \pm 0,0065$ | $78,72\% \pm 0,57\%$ | $74,71\% \pm 0,42\%$ |
| | *Full Train (20k)* | $\mathbf{0,8079}$ | $\mathbf{0,7793}$ | $\mathbf{0,6916}$ | $\mathbf{78,46\%}$ | $\mathbf{76,55\%}$ |
| **Random Forest** | 0% (Thô LLM) | $0,7763 \pm 0,0019$ | $0,7455 \pm 0,0028$ | $0,6796 \pm 0,0095$ | $75,12\% \pm 0,31\%$ | $79,54\% \pm 0,56\%$ |
| | 25% | $0,7907 \pm 0,0010$ | $0,7647 \pm 0,0010$ | $0,6940 \pm 0,0123$ | $76,72\% \pm 0,15\%$ | $79,15\% \pm 0,66\%$ |
| | 50% | $0,8031 \pm 0,0040$ | $0,7722 \pm 0,0027$ | $0,6997 \pm 0,0037$ | $77,35\% \pm 0,27\%$ | $78,75\% \pm 0,71\%$ |
| | 75% | $0,7999 \pm 0,0115$ | $0,7765 \pm 0,0044$ | $0,6983 \pm 0,0051$ | $77,72\% \pm 0,46\%$ | $77,53\% \pm 0,63\%$ |
| | 100% (Full Gold Sub) | $0,7952 \pm 0,0037$ | $0,7799 \pm 0,0041$ | $0,6904 \pm 0,0067$ | $78,02\% \pm 0,42\%$ | $75,68\% \pm 0,63\%$ |
| | *Full Train (20k)* | $\mathbf{0,8048}$ | $\mathbf{0,7722}$ | $\mathbf{0,6897}$ | $\mathbf{77,76\%}$ | $\mathbf{76,72\%}$ |

### 6.3. Quy luật khoa học rút ra từ biểu đồ Scaling:
Biểu đồ trực quan hóa chi tiết 3 đồ thị con (Dev, Test Seen, Test Unseen) đã được xuất lưu tại: [`reports/figures/ml/scaling_ablation_macro_f1.png`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/reports/figures/ml/scaling_ablation_macro_f1.png).

1. **Hiệu ứng sửa nhãn (Label Correction Effect):**
   - Trên cả 3 tập đánh giá, đường cong Macro-$F_1$ tăng trưởng gần như đơn điệu khi tỷ lệ sửa nhãn tăng từ 0% lên 50% - 75%.
   - Mức tăng ấn tượng nhất xuất hiện từ 0% lên 50%: Logistic Regression trên Dev tăng $+2,17\%$, trên Test Seen tăng $+2,45\%$, trên Test Unseen tăng $+2,60\%$.
   - Tại mức 75% đến 100%, đường cong có xu hướng tiệm cận (bão hòa), cho thấy việc phân xử và sửa 50% - 75% nhãn bất đồng đã thu được phần lớn giá trị thông tin của dữ liệu Human Gold.
2. **Khả năng tổng quát hóa trên miền unseen (Domain Generalization):**
   - Trên tập Test Seen (cùng 8 trang báo), các mô hình đạt điểm số rất sát với Dev ($\sim 0,78 - 0,79$ Macro-$F_1$).
   - Trên tập Test Unseen (4 trang báo hoàn toàn mới), xuất hiện hiện tượng domain shift khiến Macro-$F_1$ giảm xuống dải $0,66 - 0,70$. Tuy nhiên, việc huấn luyện trên Full Train 20.000 bài giúp cải thiện vượt bậc khả năng thích ứng: Logistic Regression đạt $\mathbf{0,7053}$ và Linear SVM đạt $\mathbf{0,7077}$.
3. **Quản lý Artifacts & Manifests:**
   - Toàn bộ 64 mô hình đã được lưu vết và đăng ký đầy đủ vào [`data/manifests/model_manifest.csv`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/data/manifests/model_manifest.csv).
   - Tệp mô hình nhị phân lưu tại [`models/saved_models/`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/models/saved_models/).
   - Toàn bộ 128 tệp dự đoán chi tiết trên Test Seen và Test Unseen lưu tại [`predictions/`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/predictions/).

---

## 7. Hướng dẫn sử dụng các thành phần dùng chung (Shared Components)

Toàn bộ các mô đun dùng chung đã được đặt đúng cấu trúc thư mục của repository (`src/ds_project/`), hỗ trợ gọi đơn giản từ bất kỳ kịch bản nào trong dự án.

### 7.1. Dataset Loader (`loader.py`)

Đường dẫn: [`src/ds_project/correction/loader.py`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/src/ds_project/correction/loader.py)

#### Ví dụ 1: Tải tập Train hoàn chỉnh (Full Train sau sửa nhãn)
```python
from ds_project.correction.loader import load_train_set

# Tải 20.000 dòng train đã hiệu chỉnh nhãn (subset dùng human gold, ngoài subset dùng LLM)
df_train = load_train_set(variant="full_corrected")
print(f"Train size: {len(df_train)}")  # 20,000 dòng
print(f"Nhãn: {df_train['label'].value_counts().to_dict()}")

# Hoặc tải 20.000 dòng giữ nguyên 100% nhãn thô của LLM
df_train_raw = load_train_set(variant="llm_raw")
```

#### Ví dụ 2: Tải tập Subset 6.400 bài phục vụ thí nghiệm sửa nhãn lồng nhau (Ablation Study)
```python
from ds_project.correction.loader import load_subset

# Tải subset với mức sửa 50% nhãn bất đồng trên seed 2026
df_sub_50 = load_subset(correction_level=0.50, seed=2026)
print(f"Subset size: {len(df_sub_50)}")  # 6,400 dòng

# Thí nghiệm so sánh 5 mức sửa: 0%, 25%, 50%, 75%, 100%
for level in [0.0, 0.25, 0.50, 0.75, 1.0]:
    df_sub = load_subset(correction_level=level, seed=2026)
    # Huấn luyện LR hoặc PhoBERT trên df_sub...
```

#### Ví dụ 3: Tải tập Dev và Test
```python
from ds_project.correction.loader import load_dev, load_test

# Tải Dev kèm nhãn Human Gold chuẩn (1.395 dòng đã phân xử)
df_dev = load_dev(include_gold=True)

# Tải Test Seen (1.000 dòng) và Test Unseen (1.200 dòng), kèm nhãn Gold nếu cần đánh giá
df_test_seen = load_test(kind="seen", include_gold=True)
df_test_unseen = load_test(kind="unseen", include_gold=True)
```

#### Ví dụ 4: Lấy trực tiếp ma trận đặc trưng cho ML
```python
from ds_project.correction.loader import get_train_dev_matrices

# Lấy ma trận X và vector nhãn y cho Version 3
X_train, y_train, X_dev, y_dev, pipeline = get_train_dev_matrices(
    train_variant="full_corrected",
    input_variant="v3"
)
print(f"X_train: {X_train.shape}, X_dev: {X_dev.shape}")
```

---

### 7.2. Hệ thống Manifests sửa nhãn

1. **Tệp `data/manifests/correction_manifest.csv`:**
   - **Mục đích:** Lưu vết kiểm toán (audit trail) chính xác đến từng `record_id` tại mỗi seed và mức sửa.
   - **Các trường dữ liệu:**
     - `record_id`: Định danh duy nhất của bài báo.
     - `seed`: Giá trị seed ngẫu nhiên (`2026`, `42`, `123`).
     - `correction_level`: Tỷ lệ sửa nhãn (`0.0`, `0.25`, `0.5`, `0.75`, `1.0`).
     - `label_before`: Nhãn ban đầu của LLM.
     - `label_after`: Nhãn sau khi sửa (bằng Human Gold nếu được chọn sửa, ngược lại giữ nguyên `label_before`).
     - `was_corrected`: Giá trị Boolean (`True`/`False`) xác nhận mẫu này có thực sự bị sửa đổi hay không.

2. **Tệp `data/manifests/subset_label_versions.csv`:**
   - **Mục đích:** Bảng tra cứu nhãn nhanh cho các pipeline huấn luyện ngoài.
   - **Các trường dữ liệu:** `record_id`, `seed`, `correction_level`, `label`.

---

### 7.3. Pipeline trích xuất đặc trưng (`pipeline.py`)

Đường dẫn: [`src/ds_project/features/pipeline.py`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/src/ds_project/features/pipeline.py)

Pipeline tuân thủ đầy đủ chuẩn Scikit-Learn Estimator/Transformer (`fit`, `transform`, `fit_transform`):

```python
from ds_project.features.pipeline import ClickbaitFeaturePipeline
from ds_project.correction.loader import load_train_set, load_dev
from xgboost import XGBClassifier

df_train = load_train_set("full_corrected")
df_dev = load_dev()

# Khởi tạo pipeline Version 3 (Kết hợp TF-IDF và 25 đặc trưng Warsaw)
pipeline = ClickbaitFeaturePipeline(variant="v3")

# Fit và transform trên tập train (CHỈ fit trên train)
X_train = pipeline.fit_transform(df_train)
y_train = df_train["label"]

# Transform trên tập dev / test
X_dev = pipeline.transform(df_dev)
y_dev = df_dev["label"]

# Huấn luyện mô hình XGBoost
model = XGBClassifier(n_estimators=100, max_depth=6, random_state=2026)
model.fit(X_train, y_train)

# Đánh giá
preds = model.predict(X_dev)
```

---

### 7.4. 25 Đặc trưng Stylometric Warsaw cho tiếng Việt (`stylometric.py`)

Đường dẫn: [`src/ds_project/features/stylometric.py`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/src/ds_project/features/stylometric.py)

Nếu các nhóm nghiên cứu muốn trích xuất riêng 25 đặc trưng dạng bảng (Pandas DataFrame) để làm feature importance hoặc phân tích phương sai:

```python
from ds_project.features.stylometric import WarsawStylometricExtractor
from ds_project.correction.loader import load_train_set

df_train = load_train_set()

extractor = WarsawStylometricExtractor()
# Trả về DataFrame với đúng 25 tên cột chuẩn hóa
df_features = extractor.transform_df(df_train)

print(df_features.columns.tolist())
# ['char_count', 'word_count', 'mean_word_length', 'common_words_ratio',
#  'capital_letters_ratio', 'capital_words_count', 'punctuation_ratio',
#  'bait_punctuation_count', 'non_bait_punctuation_count', 'numbers_count',
#  'pronouns_count', 'second_person_pronouns_count', 'superlatives_ratio',
#  'speculatives_count', 'bait_phrases_count', 'similarity_score',
#  'polarity_score', 'subjectivity_score', 'ttr', 'cttr', 'maas_index',
#  'hdd', 'fres', 'fkgl', 'ari']
```

---

## 8. Kết luận & Hướng phát triển tiếp theo

1. **Hoàn thiện đầy đủ yêu cầu:**
   - Đã thực hiện trọn vẹn EDA, báo cáo EDA chi tiết, thống kê bất đồng và phân bố nhãn trên tập dữ liệu đã cập nhật.
   - Đã tạo full train sau sửa nhãn (`20.000` bài), các mức sửa nhãn lồng nhau trên subset (`6.400` bài), hệ thống manifest và loader dùng chung.
   - Đã thiết lập hoàn chỉnh 3 Version Input Variants cho ML, chuẩn hóa 25 đặc trưng Warsaw tiếng Việt và chạy đối chứng thực nghiệm với đầy đủ các mô hình (XGBoost, LightGBM, Logistic Regression, Linear SVM, Random Forest).
   - Đã tối ưu siêu tham số và ngưỡng phân loại trên tập Dev, đưa cả 4 mô hình vượt mốc **0,80+ Macro-$F_1$**.
   - Đã thực thi trọn vẹn 64 lượt huấn luyện EVAL2 Scaling Ablation trên 3 tập dữ liệu (Dev, Test Seen, Test Unseen), lưu trữ đầy đủ 64 checkpoint mô hình, 128 tệp dự đoán test, manifest mô hình và biểu đồ phân tích đa phân vùng.
2. **Khuyến nghị cho các thành viên phụ trách Deep Learning (PhoBERT, XLM-R, TextCNN, RNN):**
   - Sử dụng `load_train_set(variant="full_corrected")` để nạp tập train chuẩn.
   - Sử dụng `load_subset(correction_level, seed)` để chạy ablation study theo tỷ lệ sửa nhãn.
   - Khi bước sang giai đoạn chốt kết quả (EVAL2), sử dụng `load_test("seen")` và `load_test("unseen")` để đánh giá và báo cáo điểm số cuối cùng.
