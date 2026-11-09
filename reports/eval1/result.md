# EVAL1 – Báo cáo kết quả đánh giá

> Báo cáo tổng hợp trực tiếp từ 12 tệp CSV được cung cấp. Tên cấu hình `T`, `TC`, `TS`, `TSC` được giữ nguyên như trong dữ liệu; chưa suy diễn ý nghĩa chữ viết tắt khi không có định nghĩa kèm theo.

## 1. Tóm tắt kết quả

- Tập dữ liệu DEV có **1400** bản ghi; có nhãn gold cho **1395** bản ghi và thiếu gold cho **5** bản ghi.
- Trong gold hiện có: nhãn `0` = **746**, nhãn `1` = **649**; `needs_adjudication` = **319** theo `gold_summary.csv`.
- Mức đồng thuận của người gán nhãn: raw agreement **77.71%**, Cohen's kappa **0.5511** trên **1400** cặp; có **320** trường hợp adjudication (**22.86%**) theo `human_summary(1).csv`.
- Cấu hình đứng đầu theo Macro-F1 là **TSC (0.7927)**, kế đến **TS (0.7900)**. Chênh lệch nhỏ giữa hai cấu hình này không đủ để kết luận chắc chắn từ kết quả bootstrap hiện có.
- Trên cả bốn cấu hình, số `FN` (nhãn thật `1`, dự đoán `0`) đều lớn hơn `FP` (nhãn thật `0`, dự đoán `1`); việc tăng recall cho nhãn `1` là một hướng cải thiện đáng xem xét.

## 2. Gold labels và gán nhãn thủ công

### 2.1. Phân bố dữ liệu gold
| Chỉ số | Số lượng |
| --- | --- |
| DEV records | 1400 |
| Gold labels | 1395 |
| Missing gold | 5 |
| Nhãn 0 | 746 |
| Nhãn 1 | 649 |
| Needs adjudication (gold summary) | 319 |

### 2.2. Chỉ số tổng hợp giữa người gán nhãn
| Chỉ số | Giá trị |
| --- | --- |
| Số cặp | 1400 |
| Raw agreement | 77.71% |
| Cohen's kappa | 0.5511 |
| Adjudication | 320 |
| Adjudication rate | 22.86% |
| Flag A rate | 0.64% |
| Flag B rate | 0.79% |

### 2.3. Đồng thuận theo cặp annotator
| Annotator A | Annotator B | n | Agreement | Kappa | Disagreement |
| --- | --- | --- | --- | --- | --- |
| P1 | P2 | 156 | 69.87% | 0.4346 | 30.13% |
| P1 | P9 | 156 | 83.97% | 0.6849 | 16.03% |
| P2 | P3 | 156 | 81.41% | 0.6329 | 18.59% |
| P3 | P4 | 156 | 85.90% | 0.7123 | 14.10% |
| P4 | P5 | 156 | 75.00% | 0.5081 | 25.00% |
| P6 | P5 | 155 | 69.03% | 0.3482 | 30.97% |
| P6 | P7 | 155 | 82.58% | 0.6269 | 17.42% |
| P7 | P8 | 155 | 79.35% | 0.5528 | 20.65% |
| P8 | P9 | 155 | 72.26% | 0.4638 | 27.74% |

**Lưu ý kiểm tra dữ liệu:** `gold_summary.csv` ghi nhận `needs_adjudication = 319`, trong khi `human_summary(1).csv` ghi nhận `adjudication_n = 320`. Báo cáo giữ nguyên cả hai con số; cần kiểm tra định nghĩa tập đếm hoặc quy trình lọc nếu hai trường được kỳ vọng trùng nhau.

## 3. Coverage và tính hợp lệ của output

| Cấu hình | DEV | Returned | Valid | Invalid | Missing | Coverage | Retry records | No retry (valid) | Common no fallback | Evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T | 1400 | 1400 | 1399 | 1 | 0 | 99.93% | 2 | 1398 | 1398 | 1394 |
| TC | 1400 | 1400 | 1399 | 1 | 0 | 99.93% | 2 | 1398 | 1398 | 1394 |
| TS | 1400 | 1400 | 1399 | 1 | 0 | 99.93% | 3 | 1397 | 1398 | 1394 |
| TSC | 1400 | 1400 | 1398 | 2 | 0 | 99.86% | 4 | 1396 | 1398 | 1393 |

- `coverage` được giữ theo giá trị do file gốc cung cấp và hiển thị dưới dạng phần trăm.
- Số bản ghi `evaluated_n` nhỏ hơn `valid_n` vì có **5 bản ghi thiếu gold**. Với `T`, `TC`, `TS`: 1399 valid trừ 5 thiếu gold còn 1394; với `TSC`: 1398 valid trừ 5 còn 1393.
- Các cột `retry_record_n`, `valid_no_retry_n`, `common_no_fallback_n` chỉ phản ánh số đếm từ CSV; báo cáo không tự gán ý nghĩa quy trình vượt quá tên trường.

## 4. Chất lượng dự đoán LLM

### 4.1. Chỉ số tổng quan
| Cấu hình | n | Accuracy | Macro-F1 | Kappa | F1 (0) | F1 (1) | Recall (1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T | 1394 | 77.98% | 0.7707 | 0.5492 | 0.8163 | 0.7252 | 62.50% |
| TC | 1394 | 77.91% | 0.7723 | 0.5494 | 0.8115 | 0.7331 | 65.28% |
| TS | 1394 | 79.48% | 0.7900 | 0.5828 | 0.8219 | 0.7580 | 69.14% |
| TSC | 1393 | 79.68% | 0.7927 | 0.5874 | 0.8221 | 0.7632 | 70.48% |

### 4.2. Precision / recall từng nhãn
| Cấu hình | Precision 0 | Recall 0 | Precision 1 | Recall 1 |
| --- | --- | --- | --- | --- |
| T | 73.73% | 91.42% | 86.35% | 62.50% |
| TC | 74.66% | 88.87% | 83.60% | 65.28% |
| TS | 76.74% | 88.47% | 83.90% | 69.14% |
| TSC | 77.40% | 87.67% | 83.21% | 70.48% |

### 4.3. Ma trận nhầm lẫn dạng số đếm
| Cấu hình | TN | FP | FN | TP | Tổng lỗi (FP + FN) |
| --- | --- | --- | --- | --- | --- |
| T | 682 | 64 | 243 | 405 | 307 |
| TC | 663 | 83 | 225 | 423 | 308 |
| TS | 660 | 86 | 200 | 448 | 286 |
| TSC | 654 | 92 | 191 | 456 | 283 |

Trong bảng: `TN` = nhãn 0 dự đoán 0; `FP` = nhãn 0 dự đoán 1; `FN` = nhãn 1 dự đoán 0; `TP` = nhãn 1 dự đoán 1. Các mã `0` và `1` chưa được ánh xạ sang tên nhãn nghiệp vụ vì dữ liệu nguồn không mô tả chúng.

**Nhận xét:** Khi so sánh `T` với `TSC`, Macro-F1 tăng từ **0.7707** lên **0.7927**. Recall nhãn `1` tăng từ **62.50%** lên **70.48%**, đồng thời FP tăng từ **64** lên **92**. Hai cấu hình có số mẫu đánh giá khác nhau (`1394` và `1393`), vì vậy khi kết luận về cải thiện theo cặp nên ưu tiên kết quả trên tập chung ở mục 5.

## 5. So sánh cặp bằng paired bootstrap

Giá trị `Δ Macro-F1` bằng **Macro-F1(B) − Macro-F1(A)** trên tập bản ghi chung của từng cặp. Cột CI sao chép từ file nguồn (mức tin cậy không được nêu trong CSV).

| A | B | n chung | F1(A) | F1(B) | Δ Macro-F1 | CI thấp | CI cao | CI chứa 0? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T | TC | 1394 | 0.7707 | 0.7723 | 0.0016 | -0.0095 | 0.0134 | Có |
| T | TS | 1394 | 0.7707 | 0.7900 | 0.0193 | 0.0053 | 0.0340 | Không |
| T | TSC | 1393 | 0.7705 | 0.7927 | 0.0222 | 0.0063 | 0.0377 | Không |
| TC | TS | 1394 | 0.7723 | 0.7900 | 0.0177 | 0.0056 | 0.0301 | Không |
| TC | TSC | 1393 | 0.7721 | 0.7927 | 0.0206 | 0.0068 | 0.0338 | Không |
| TS | TSC | 1393 | 0.7898 | 0.7927 | 0.0029 | -0.0070 | 0.0125 | Có |

- `T → TS` và `T → TSC` có chênh lệch dương với CI **không chứa 0** trong kết quả đã cung cấp.
- `TC → TS` và `TC → TSC` cũng có CI **không chứa 0**.
- `T → TC` và `TS → TSC` có CI **chứa 0**; chưa có đủ bằng chứng từ các khoảng này để khẳng định B tốt hơn A.

## 6. Hiệu năng theo nguồn và chuyên mục

Dưới đây là **Macro-F1** theo từng nhóm trong `metrics_by_group.csv`. `n (T)` và `n (TSC)` là cỡ mẫu tương ứng của hai cấu hình; không suy diễn n của các cấu hình khác từ hai cột này. Nhóm ít mẫu (ví dụ `Xe`, n = 11) có chỉ số dễ dao động.

### 6.1. Theo nguồn tin
| Nguồn | n (T) | n (TSC) | T | TC | TS | TSC |
| --- | --- | --- | --- | --- | --- | --- |
| 2sao | 175 | 175 | 0.8076 | 0.8065 | 0.8109 | 0.8097 |
| afamily | 174 | 174 | 0.7467 | 0.7341 | 0.7287 | 0.7368 |
| cafef | 175 | 175 | 0.6956 | 0.6753 | 0.7412 | 0.7367 |
| dantri | 174 | 174 | 0.6390 | 0.6878 | 0.7138 | 0.6994 |
| eva | 173 | 173 | 0.6855 | 0.7121 | 0.7383 | 0.7546 |
| kenh14 | 174 | 173 | 0.7279 | 0.7212 | 0.7485 | 0.7466 |
| thanhnien | 174 | 174 | 0.6821 | 0.7139 | 0.7196 | 0.7138 |
| tuoitre | 175 | 175 | 0.6613 | 0.6557 | 0.6212 | 0.6547 |

### 6.2. Theo chuyên mục
| Chuyên mục | n (T) | n (TSC) | T | TC | TS | TSC |
| --- | --- | --- | --- | --- | --- | --- |
| Bất động sản | 30 | 30 | 0.7115 | 0.5957 | 0.7115 | 0.6732 |
| Công nghệ - Khoa học | 47 | 47 | 0.6398 | 0.6357 | 0.6083 | 0.5890 |
| Du lịch - Ẩm thực | 43 | 43 | 0.6742 | 0.7171 | 0.7171 | 0.7832 |
| Giáo dục | 57 | 57 | 0.7799 | 0.7632 | 0.7467 | 0.7905 |
| Giải trí - Văn hóa | 152 | 152 | 0.7476 | 0.7662 | 0.7653 | 0.7933 |
| Khác | 22 | 22 | 0.8706 | 0.9394 | 0.9394 | 0.9394 |
| Kinh tế - Tài chính | 141 | 141 | 0.7208 | 0.7063 | 0.7389 | 0.7032 |
| Làm đẹp - Thời trang | 64 | 64 | 0.7263 | 0.7733 | 0.8333 | 0.8333 |
| Pháp luật | 35 | 35 | 0.4615 | 0.4697 | 0.4776 | 0.4776 |
| Sức khỏe | 81 | 81 | 0.7530 | 0.7527 | 0.7890 | 0.7769 |
| Thế giới | 88 | 88 | 0.7203 | 0.7203 | 0.7325 | 0.7326 |
| Thể thao | 70 | 70 | 0.5868 | 0.5758 | 0.5837 | 0.5837 |
| Thời sự - Xã hội | 287 | 287 | 0.7258 | 0.7377 | 0.7415 | 0.7339 |
| Tử vi - Phong thủy | 41 | 41 | 0.5632 | 0.4459 | 0.4533 | 0.4605 |
| Xe | 11 | 11 | 0.4762 | 0.4500 | 0.4211 | 0.4500 |
| Đời sống - Gia đình | 225 | 224 | 0.6795 | 0.6732 | 0.7131 | 0.7115 |

**Điểm cần chú ý:** Ở `TSC`, các nhóm có Macro-F1 thấp theo *nguồn* gồm `tuoitre` và `dantri`; theo *chuyên mục* gồm `Xe`, `Tử vi - Phong thủy` và `Pháp luật`. Các nhóm này đáng ưu tiên kiểm tra thêm nhưng cần cân nhắc số lượng bản ghi và phân bố nhãn.

## 7. Phân tích các dự đoán sai

### 7.1. Tổng số lỗi theo cấu hình
| Cấu hình | Số dòng lỗi | FP (0→1) | FN (1→0) |
| --- | --- | --- | --- |
| T | 307 | 64 | 243 |
| TC | 308 | 83 | 225 |
| TS | 286 | 86 | 200 |
| TSC | 283 | 92 | 191 |

Các con số khớp với cột `FP`/`FN` từ `llm_metrics.csv`. Toàn bộ bản ghi lỗi vẫn được giữ trong bốn file `errors_*.csv`, không đưa tất cả vào Markdown để báo cáo dễ đọc.

### 7.2. Ví dụ lỗi từ TSC
| Loại lỗi | Nguồn | Chuyên mục | Tiêu đề |
| --- | --- | --- | --- |
| FN | kenh14 | Thời sự - Xã hội | Người phụ nữ 40 tuổi đến ngân hàng Agribank rút tiền tiết kiệm, nhân viên phát hiện dấu hiệu lạ lập tức báo công an |
| FN | eva | Thời sự - Xã hội | Hoa sữa nở nồng nặc, người dân có được tự ý cắt tỉa cây trước nhà? |
| FN | kenh14 | Kinh tế - Tài chính | Giá vàng 9999 SJC hôm nay 23/9/2026 bao nhiêu 1 chỉ? |
| FP | afamily | Giải trí - Văn hóa | Tình trạng của Triệu Lệ Dĩnh |
| FP | 2sao | Đời sống - Gia đình | Ảnh nét căng: Văn Thanh mang trap xịn xò đến nhà vợ 'trâm anh thế phiệt' dạm ngõ, trao ngay 'thẻ lương' uy tín |
| FP | thanhnien | Đời sống - Gia đình | Thanh Niên: Từ tờ báo 1.300 đồng đến cú click chuột |

### 7.3. Nhóm lỗi nhiều nhất trong TSC
**Theo nguồn:** `cafef` (46), `kenh14` (43), `eva` (41), `afamily` (39), `thanhnien` (34).

**Theo chuyên mục:** `Đời sống - Gia đình` (53), `Thời sự - Xã hội` (47), `Kinh tế - Tài chính` (37), `Giải trí - Văn hóa` (27), `Thể thao` (21).

> **Lưu ý:** Số lỗi tuyệt đối cao không đồng nghĩa tỷ lệ lỗi cao nếu mỗi nhóm có cỡ mẫu khác nhau. Để so sánh khách quan giữa các nhóm, xem Macro-F1, Accuracy và `n` trong bảng mục 6.

## 8. Các bản ghi thiếu gold

Tổng số: **5**.
| Nguồn | Chuyên mục | Tiêu đề |
| --- | --- | --- |
| afamily | Giải trí - Văn hóa | Hòa Minzy đáp trả |
| eva | Sức khỏe | Loại quả lọt top tốt trên thế giới trồng thành công ở Việt Nam, là đặc sản nổi tiếng 200.000 đ/kg, tốt cho sức khỏe |
| eva | Đời sống - Gia đình | 10 câu đố vui lịch sử Việt Nam nhân dịp Lễ 2/9, bé làm đúng được bao nhiêu câu? |
| kenh14 | Sức khỏe | Nghiên cứu của ĐH Mỹ: Ăn 2 LOẠI trái cây giúp hạ huyết áp, cải thiện mạch máu và "bảo vệ" thận, ở Việt Nam bán rẻ bèo |
| thanhnien | Thời sự - Xã hội | Thêm chỗ dựa cho chắc |

Các bản ghi này nên được rà soát nhãn nếu muốn tăng độ bao phủ của tập đánh giá; không tự điền nhãn khi chưa có adjudication/ground truth được xác nhận.

## 9. Kết luận và bước tiếp theo

1. **Ưu tiên `TSC` nếu mục tiêu chỉ là Macro-F1 quan sát được:** đạt **0.7927** với **1393** mẫu đánh giá, nhưng hơn `TS` rất ít (**Δ = 0.0029** trên 1393 bản ghi chung); CI của chênh lệch này chứa 0.
2. **`TS` là lựa chọn gần tương đương về kết quả trong tập hiện có:** Macro-F1 **0.7900**, `invalid_n = 1` và `retry_record_n = 3`, so với `TSC` có `invalid_n = 2`, `retry_record_n = 4`; chưa có dữ liệu chi phí/độ trễ để kết luận về hiệu quả vận hành.
3. **Rà soát độ tin cậy của gold:** xử lý 5 bản ghi thiếu nhãn và xác minh sự khác nhau giữa 319 và 320 bản ghi adjudication ở hai file thống kê.
4. **Phân tích sâu lỗi theo nhóm:** tập trung vào các chuyên mục có Macro-F1 thấp và các lỗi nhãn `1` bị dự đoán thành `0`, đồng thời kiểm tra phân bố nhãn để tránh kết luận sai khi dữ liệu mất cân bằng.

## 10. Danh sách tệp nguồn

Các đường dẫn dưới đây là đường dẫn **tương đối**; chúng hoạt động trên GitHub khi file `.md` và các CSV nằm trong cùng thư mục và tên file được giữ nguyên.

- [`coverage.csv`](./coverage.csv)
- [`gold_summary.csv`](./gold_summary.csv)
- [`human_agreement(1).csv`](./human_agreement%281%29.csv)
- [`human_summary(1).csv`](./human_summary%281%29.csv)
- [`llm_metrics.csv`](./llm_metrics.csv)
- [`metrics_by_group.csv`](./metrics_by_group.csv)
- [`paired_bootstrap(1).csv`](./paired_bootstrap%281%29.csv)
- [`missing_dev_gold.csv`](./missing_dev_gold.csv)
- [`errors_T.csv`](./errors_T.csv)
- [`errors_TC.csv`](./errors_TC.csv)
- [`errors_TS.csv`](./errors_TS.csv)
- [`errors_TSC.csv`](./errors_TSC.csv)

---
**Nguồn số liệu:** 12 CSV được cung cấp. Các số thập phân trình bày có làm tròn; khi cần dùng giá trị chính xác, tham chiếu file CSV gốc.
