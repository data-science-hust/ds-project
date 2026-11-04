# Báo Cáo Thực Nghiệm Pipeline ML, Kiến Trúc 3 Phiên Bản & Hướng Dẫn Sử Dụng Thành Phần Dùng Chung

---

## Mục lục
1. [Tổng quan & Mục tiêu thực nghiệm](#1-tổng-quan--mục-tiêu-thực-nghiệm)
2. [Giải đáp: Dữ liệu huấn luyện & Tại sao đánh giá trên tập Dev](#2-giải-đáp-dữ-liệu-huấn-luyện--tại-sao-đánh-giá-trên-tập-dev)
3. [Kiến trúc 3 Phiên bản Dữ liệu đầu vào (Input Variants)](#3-kiến-trúc-3-phiên-bản-dữ-liệu-đầu-vào-input-variants)
4. [Bảng kết quả so sánh đối chứng mô hình (kèm XGBoost)](#4-bảng-kết-quả-so-sánh-đối-chứng-mô-hình-kèm-xgboost)
5. [Hướng dẫn sử dụng các thành phần dùng chung (Shared Components)](#5-hướng-dẫn-sử-dụng-các-thành-phần-dùng-chung-shared-components)
   - [5.1. Dataset Loader (`loader.py`)](#51-dataset-loader-loaderpy)
   - [5.2. Hệ thống Manifests sửa nhãn](#52-hệ-thống-manifests-sửa-nhãn)
   - [5.3. Pipeline trích xuất đặc trưng (`pipeline.py`)](#53-pipeline-trích-xuất-đặc-trưng-pipelinepy)
   - [5.4. 25 Đặc trưng Stylometric Warsaw cho tiếng Việt (`stylometric.py`)](#54-25-đặc-trưng-stylometric-warsaw-cho-tiếng-việt-stylometricpy)

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
| **Version 1** (Baseline TF-IDF) | Logistic Regression | 35.000 | 77,49% | 0,7679 | 0,7274 |
| **Version 1** (Baseline TF-IDF) | Linear SVM | 35.000 | 77,42% | 0,7673 | 0,7273 |
| **Version 1** (Baseline TF-IDF) | Random Forest | 35.000 | 70,32% | 0,6689 | 0,5624 |
| **Version 1** (Baseline TF-IDF) | **XGBoost** | 35.000 | 75,63% | 0,7476 | 0,7007 |
| **Version 1** (Baseline TF-IDF) | **LightGBM** | 35.000 | 75,27% | 0,7391 | 0,6797 |
| **Version 2** (25 Warsaw Features) | Logistic Regression | 25 | 72,69% | 0,7105 | 0,6416 |
| **Version 2** (25 Warsaw Features) | Linear SVM | 25 | 72,33% | 0,7053 | 0,6324 |
| **Version 2** (25 Warsaw Features) | Random Forest | 25 | 73,69% | 0,7233 | 0,6618 |
| **Version 2** (25 Warsaw Features) | **XGBoost** | 25 | 72,54% | 0,7144 | 0,6583 |
| **Version 2** (25 Warsaw Features) | **LightGBM** | 25 | 74,62% | 0,7350 | 0,6805 |
| **Version 3** (Combined Hybrid) | **Logistic Regression** | **35.025** | **78,42%** | **0,7779** | **0,7403** |
| **Version 3** (Combined Hybrid) | **Linear SVM** | **35.025** | **77,71%** | **0,7706** | **0,7321** |
| **Version 3** (Combined Hybrid) | Random Forest | 35.025 | 71,90% | 0,6929 | 0,6032 |
| **Version 3** (Combined Hybrid) | **XGBoost** | **35.025** | **76,63%** | **0,7581** | **0,7135** |
| **Version 3** (Combined Hybrid) | **LightGBM** | **35.025** | **76,20%** | **0,7515** | **0,7004** |

### Nhận xét & Đánh giá khoa học:
1. **Chứng minh giả thuyết nghiên cứu (RQ2):**
   - Trên **TẤT CẢ 5 MÔ HÌNH**, **Version 3** đều cho kết quả vượt trội hơn so với Version 1 thuần TF-IDF.
   - Logistic Regression tăng từ $0,7679 \to \mathbf{0,7779}$ (+1,00% Macro-$F_1$, +1,29% Clickbait-$F_1$).
   - Linear SVM tăng từ $0,7673 \to \mathbf{0,7706}$ (+0,33% Macro-$F_1$).
   - Random Forest tăng từ $0,6689 \to \mathbf{0,6929}$ (+2,40% Macro-$F_1$).
   - **XGBoost tăng từ $0,7476 \to \mathbf{0,7581}$ (+1,05% Macro-$F_1$, +1,28% Clickbait-$F_1$)**.
   - **LightGBM tăng từ $0,7391 \to \mathbf{0,7515}$ (+1,24% Macro-$F_1$, +2,07% Clickbait-$F_1$)**.
2. **Sức mạnh của 25 đặc trưng Warsaw (Version 2):**
   - Chỉ với **25 chiều**, các mô hình dạng cây (LightGBM, Random Forest, XGBoost) đạt Macro-$F_1$ từ **0,7144 đến 0,7350**, vượt trội hoàn toàn so với việc Random Forest chạy trên không gian 35.000 chiều TF-IDF ($0,6689$).
   - Điều này khẳng định 25 đặc trưng ngữ phong chứa hàm lượng thông tin cấu trúc rất cô đọng, hỗ trợ đắc lực cho các bộ phân loại dạng cây (Tree-based models).

---

## 4. Tối ưu hóa siêu tham số (Hyperparameter Tuning) và Ngưỡng quyết định (Decision Threshold) trên tập Dev

Theo đúng nguyên tắc khoa học (§8.1 trong Đề cương), tập kiểm định `dev` được dùng để lựa chọn tham số tối ưu và xác định ngưỡng phân loại (decision threshold $\tau^*$). Việc tối ưu được tiến hành trên cấu hình đầu vào tốt nhất: **Version 3** (Combined TF-IDF + 25 đặc trưng Warsaw), huấn luyện trên tập **Full Train Corrected (20.000 mẫu)** và tinh chỉnh trên **Dev Gold (1.395 mẫu)**.

Kịch bản tối ưu hóa: [`scripts/tune_hyperparameters.py`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/scripts/tune_hyperparameters.py) và [`scripts/tune_rf.py`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/scripts/tune_rf.py).  
Dữ liệu chi tiết từng cấu hình được lưu tại: [`reports/hyperparameter_tuning_results.csv`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/reports/hyperparameter_tuning_results.csv).  
Cấu hình tối ưu chuẩn JSON được lưu tại: [`reports/best_hyperparameters.json`](file:///d:/SOICT/IT4142E%20-%20DS/ds-project/reports/best_hyperparameters.json).

### 4.1. Bảng đối chiếu kết quả Trước và Sau tối ưu trên 4 mô hình:

| Mô hình | Trạng thái | Siêu tham số chính | Ngưỡng $\tau^*$ | Accuracy | Macro-$F_1$ | Clickbait $F_1$ | Clickbait Recall | Clickbait Precision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Trước tối ưu (Default) | $C=1.0$, `lbfgs` | 0,50 | 78,42% | 0,7779 | 0,7403 | 71,03% | 77,29% |
| | **Sau tối ưu (Tuned)** | **$C=0.5$, `lbfgs`, unweighted** | **0,34** | **80,50%** | **0,8045** | **0,7946** | **81,05%** | **77,93%** |
| | *Mức độ cải thiện* | *Regularization mạnh hơn + threshold tuning* | | *+2,08%* | *+2,66%* | *+5,43%* | *+10,02%* | *+0,64%* |
| **Linear SVM** | Trước tối ưu (Default) | $\alpha=10^{-4}$, `loss=hinge` | 0,00 | 77,71% | 0,7706 | 0,7321 | 70,26% | 76,43% |
| | **Sau tối ưu (Tuned)** | **$\alpha=5 \times 10^{-4}$, `loss=hinge`** | **-0,50** | **79,71%** | **0,7965** | **0,7851** | **79,66%** | **77,40%** |
| | *Mức độ cải thiện* | *Tăng $\alpha$ giảm overfit + dịch margin decision* | | *+2,00%* | *+2,59%* | *+5,30%* | *+9,40%* | *+0,97%* |
| **Random Forest** | Trước tối ưu (Default) | $N=200$, `max_depth=25` | 0,50 | 71,90% | 0,6929 | 0,6032 | 50,85% | 74,18% |
| | **Sau tối ưu (Tuned)** | **$N=200$, `max_depth=70`, `min_split=5`** | **0,33** | **78,92%** | **0,7892** | **0,7851** | **82,74%** | **74,69%** |
| | *Mức độ cải thiện* | *Tăng chiều sâu cây (25 $\to$ 70) + threshold tuning* | | *+7,02%* | *+9,63%* | *+18,19%* | *+31,89%* | *+0,51%* |
| **XGBoost** | Trước tối ưu (Default) | `depth=6`, $\eta=0.1$, $N=150$ | 0,50 | 76,63% | 0,7581 | 0,7135 | 67,03% | 76,27% |
| | **Sau tối ưu (Tuned)** | **`depth=6`, $\eta=0.15$, $N=150$, `colsample=0.8`** | **0,41** | **78,78%** | **0,7843** | **0,7566** | **70,88%** | **81,13%** |
| | *Mức độ cải thiện* | *Tăng learning rate + subsampling + threshold* | | *+2,15%* | *+2,62%* | *+4,31%* | *+3,85%* | *+4,86%* |

### 4.2. Nhận xét kỹ thuật về quá trình tối ưu:
1. **Hiện tượng nén xác suất và vai trò của Ngưỡng quyết định ($\tau^*$):**
   - Với các bộ dữ liệu văn bản thưa nhiều chiều (35.000 từ + ngữ phong), các mô hình máy học (đặc biệt là Random Forest và XGBoost) thường đưa ra phân phối xác suất dự đoán tập trung quanh giá trị $0,35 - 0,45$, khiến ngưỡng mặc định $0,50$ bỏ sót rất nhiều tin clickbait (Recall clickbait chỉ đạt $50,85\%$ ở RF và $67,03\%$ ở XGBoost).
   - Khi quét ngưỡng quyết định trên tập Dev, ngưỡng tối ưu $\tau^*$ rơi vào khoảng $0,33 - 0,41$. Việc hiệu chỉnh ngưỡng này giải phóng triệt để độ nhạy của mô hình, giúp Recall của clickbait tăng vọt lên trên $80\%$ mà Precision vẫn được giữ vững quanh mức $75 - 81\%$.
2. **Đột phá của Random Forest:**
   - Trong thí nghiệm baseline, Random Forest bị giới hạn ở `max_depth=25`, khiến cây quyết định không đủ độ sâu để kết hợp đồng thời các n-gram từ/ký tự thưa với 25 đặc trưng ngữ phong Warsaw, dẫn đến F1 clickbait chỉ đạt $0,6032$.
   - Khi tăng độ sâu lên `max_depth=70` kết hợp ngưỡng $\tau^*=0,33$, Random Forest có bước nhảy vọt ngoạn mục nhất trong tất cả các mô hình: **Macro-$F_1$ tăng từ $0,6929 \to 0,7892$ (+9,63%)**, vượt qua cả XGBoost ($0,7843$) và vươn lên cạnh tranh sát nút với Linear SVM ($0,7965$).
3. **Mô hình xuất sắc nhất toàn diện:**
   - **Logistic Regression (Tuned)** đạt hiệu năng cao nhất trên tập Dev với **Macro-$F_1 = 0,8045$**, **Accuracy = $80,50\%$** và **Clickbait-$F_1 = 0,7946$** (với $C=0.5$ và $\tau^*=0,34$). Kết quả này vượt qua cả hiệu năng zero-shot ban đầu của mô hình LLM lớn Gemma TSC ($0,7927$), chứng minh sức mạnh của phương pháp cổ điển được chuẩn hóa và tối ưu khoa học.

---

## 5. Hướng dẫn sử dụng các thành phần dùng chung (Shared Components)

Toàn bộ các mô đun dùng chung đã được đặt đúng cấu trúc thư mục của repository (`src/ds_project/`), hỗ trợ gọi đơn giản từ bất kỳ kịch bản nào trong dự án.

### 5.1. Dataset Loader (`loader.py`)

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

# Tải Test Seen (1.000 dòng) và Test Unseen (1.200 dòng)
df_test_seen = load_test(kind="seen")
df_test_unseen = load_test(kind="unseen")
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

### 5.2. Hệ thống Manifests sửa nhãn

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

### 5.3. Pipeline trích xuất đặc trưng (`pipeline.py`)

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

### 5.4. 25 Đặc trưng Stylometric Warsaw cho tiếng Việt (`stylometric.py`)

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

## 6. Kết luận & Hướng phát triển tiếp theo

1. **Hoàn thiện đầy đủ yêu cầu:**
   - Đã thực hiện trọn vẹn EDA, báo cáo EDA chi tiết, thống kê bất đồng và phân bố nhãn.
   - Đã tạo full train sau sửa nhãn (`20.000` bài), các mức sửa nhãn lồng nhau trên subset (`6.400` bài), hệ thống manifest và loader dùng chung.
   - Đã thiết lập hoàn chỉnh 3 Version Input Variants cho ML, chuẩn hóa 25 đặc trưng Warsaw tiếng Việt và chạy đối chứng thực nghiệm với đầy đủ các mô hình (XGBoost, Logistic Regression, Linear SVM, Random Forest).
2. **Khuyến nghị cho các thành viên phụ trách Deep Learning (PhoBERT, XLM-R, TextCNN, RNN):**
   - Sử dụng `load_train_set(variant="full_corrected")` để nạp tập train chuẩn.
   - Sử dụng `load_subset(correction_level, seed)` để chạy ablation study theo tỷ lệ sửa nhãn.
   - Khi bước sang giai đoạn chốt kết quả (EVAL2), sử dụng `load_test("seen")` và `load_test("unseen")` để đánh giá và báo cáo điểm số cuối cùng.
