# Báo cáo dữ liệu

Tạo lúc 2026-10-23T02:51:37+07:00 bởi `scripts/build_clean.py`.

## 1. Crawl thô

Nguồn: `reports/crawl_stats.csv` (do `scripts/export_raw.py` ghi cùng `data/raw/articles_raw.jsonl`).

- Đã chọn vào quota thô: **30,000/30,000**; thiếu **0**.
- Năm đăng: 2026: 30,000.
- Lượt thử URL bài 39,798; tải được (có title) 33,562; success/partial không title 3,891.
- `extraction_status` trong raw: `success` 29,467, `partial` 533.
- Thiếu sapo sau crawl 839; lấy lại bằng refetch 310; còn thiếu trong raw **529** (1.76%). Thiếu category 4.

| Nguồn | Nhóm | Run | Inventory URL | Lượt thử | Tải được | Không title | Đã chọn | Target | Thiếu | Thiếu sapo (crawl) | Refetch | Thiếu sapo (raw) | Thiếu category | Category gốc | Ngày đăng |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| afamily | seen | `metadata_2024_20261021` | 202,952 | 3,505 | 3,500 | 3 | 3,500 | 3,500 | 0 | 4 | 0 | 4 | 0 | 18 | 2026-01-01 → 2026-10-20 |
| 2sao | seen | `metadata_2024_20261021` | 116,721 | 7,897 | 3,720 | 3,739 | 3,500 | 3,500 | 0 | 1 | 0 | 1 | 0 | 9 | 2026-01-01 → 2026-10-21 |
| eva | seen | `metadata_fix_20261023` | 76,660 | 3,506 | 3,500 | 2 | 3,500 | 3,500 | 0 | 159 | 0 | 159 | 0 | 16 | 2026-01-01 → 2026-10-21 |
| kenh14 | seen | `metadata_2024_20261021` | 302,704 | 3,503 | 3,500 | 3 | 3,500 | 3,500 | 0 | 11 | 0 | 11 | 4 | 22 | 2026-01-01 → 2026-10-21 |
| cafef | seen | `metadata_2024_20261021` | 331,616 | 3,514 | 3,500 | 6 | 3,500 | 3,500 | 0 | 2 | 0 | 2 | 0 | 13 | 2026-01-01 → 2026-10-21 |
| thanhnien | seen | `metadata_2024_20261021` | 271,331 | 3,519 | 3,500 | 7 | 3,500 | 3,500 | 0 | 8 | 0 | 8 | 0 | 105 | 2026-01-01 → 2026-10-21 |
| tuoitre | seen | `metadata_2024_20261021` | 210,518 | 3,509 | 3,500 | 6 | 3,500 | 3,500 | 0 | 3 | 0 | 3 | 0 | 53 | 2026-01-01 → 2026-10-21 |
| dantri | seen | `metadata_2024_20261021` | 214,992 | 5,479 | 3,501 | 100 | 3,500 | 3,500 | 0 | 335 | 310 | 25 | 0 | 129 | 2026-01-01 → 2026-10-21 |
| saostar | unseen | `metadata_2024_20261021` | 107,529 | 504 | 500 | 4 | 500 | 500 | 0 | 0 | 0 | 0 | 0 | 13 | 2026-01-01 → 2026-10-21 |
| yan | unseen | `metadata_fix_20261023` | 3,097 | 2,181 | 2,181 | 0 | 500 | 500 | 0 | 283 | 0 | 283 | 0 | 7 | 2026-01-02 → 2026-09-27 |
| vietnamnet | unseen | `metadata_fix_20261023` | 344,788 | 2,181 | 2,160 | 21 | 500 | 500 | 0 | 6 | 0 | 6 | 0 | 74 | 2026-01-01 → 2026-10-20 |
| nhandan | unseen | `metadata_recovery_nhandan_20261022` | 32,462 | 500 | 500 | 0 | 500 | 500 | 0 | 27 | 0 | 27 | 0 | 188 | 2026-01-27 → 2026-10-21 |

Lý do chọn/không chọn trên mọi lượt thử (`selection_status` của crawler):

| Nguồn | primary | fallback | not_article_page | missing_title | extraction_error | publication_date_unknown | before_fallback_date |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| afamily | 3,500 | 0 | 0 | 3 | 2 | 0 | 0 |
| 2sao | 3,500 | 79 | 3,739 | 0 | 438 | 140 | 1 |
| eva | 3,500 | 0 | 0 | 2 | 4 | 0 | 0 |
| kenh14 | 3,500 | 0 | 0 | 3 | 0 | 0 | 0 |
| cafef | 3,500 | 0 | 0 | 6 | 8 | 0 | 0 |
| thanhnien | 3,500 | 0 | 2 | 5 | 12 | 0 | 0 |
| tuoitre | 3,500 | 0 | 0 | 6 | 3 | 0 | 0 |
| dantri | 3,500 | 0 | 0 | 100 | 1,878 | 1 | 0 |
| saostar | 500 | 0 | 4 | 0 | 0 | 0 | 0 |
| yan | 500 | 1,243 | 0 | 0 | 0 | 0 | 438 |
| vietnamnet | 500 | 298 | 1 | 20 | 0 | 95 | 1,267 |
| nhandan | 500 | 0 | 0 | 0 | 0 | 0 | 0 |

### Chính sách và quyết định

- Quota thô: 3.500 bài/báo seen, 500 bài/báo unseen; ưu tiên 01/01/2026–21/10/2026, chỉ lùi tới 01/01/2024 khi thiếu.
- Inventory duyệt hết sitemap/feed rồi xáo trộn với seed 2026; không đặt quota theo category.
- Chỉ lưu 9 trường của plan; không lưu body/HTML/media. Thiếu sapo/category vẫn giữ (partial), không lấy SEO description.
- CafeF: timestamp không offset được coi là UTC+07:00 (nhóm đã duyệt).
- VietNamNet: cách ly 13 bản ghi sitemap (11 URL năm 2022) có ký tự điều khiển trong URL; không sửa URL nào.
- Bỏ qua sitemap tag (`sitemap-tags-NN.xml`) và URL trang tag `-tagNNN.html` (2sao, VietNamNet).
- Dân Trí: định dạng 'Dân trí 360' (video ngắn render bằng JS, không có title/ngày trong HTML) và 5 bài emagazine có tiêu đề dạng ảnh bị loại có hệ thống (quyết định của nhóm), thể hiện ở missing_title.
- Dân Trí: sapo của bài emagazine được tải lại riêng (refetch) với selector mới, chỉ nhận khi title tải lại trùng title đã lưu. `scripts/export_raw.py` điền sapo này vào `articles_raw.jsonl` (chỉ dòng chưa có sapo) và tính lại `extraction_status` theo đúng quy tắc của crawler (`success` khi đủ title, sapo, category, published_at; còn lại `partial`). File crawl gốc trong `data/crawl_work/` không bị sửa; danh sách record_id được điền nằm ở `sapo_refetch/<refetch_id>/sapo_supplement.jsonl` (status `sapo_recovered`) trong thư mục run.
- Dân Trí bị giới hạn ~100 request/3 phút (HTTP 428): delay 2,5 s và chờ ≥120 s khi gặp 428.
- YAN không có phần sapo riêng; chỉ nhận đoạn đầu được in đậm toàn bộ là sapo.

### Giới hạn

- Quota thô không phải quota sạch: xem phần Clean (checking, bỏ trùng, funnel).
- Inventory duyệt hết chỉ có nghĩa đã duyệt hết endpoint tìm được, không chứng minh bao phủ mọi bài đã đăng.

## 2. Clean

Input: `data/raw/articles_raw.jsonl`; cột `source_run_id` của parquet lấy theo `run_id` trong `reports/crawl_stats.csv`.

### Funnel: tải được → hợp lệ → sau bỏ trùng → có sapo

| Nguồn | Nhóm | Tải được | Không title* | Đã chọn | Hợp lệ | Loại | Loại do trùng | Sau bỏ trùng | Có sapo | Mục tiêu sạch | Thiếu |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| afamily | seen | 3,500 | 3 | 3,500 | 3,500 | 0 | 107 | 3,393 | 3,389 | 2,800 | 0 |
| 2sao | seen | 3,720 | 3,739 | 3,500 | 3,500 | 0 | 64 | 3,436 | 3,435 | 2,800 | 0 |
| eva | seen | 3,500 | 2 | 3,500 | 3,341 | 159 | 22 | 3,319 | 3,319 | 2,800 | 0 |
| kenh14 | seen | 3,500 | 3 | 3,500 | 3,500 | 0 | 61 | 3,439 | 3,428 | 2,800 | 0 |
| cafef | seen | 3,500 | 6 | 3,500 | 3,487 | 13 | 72 | 3,415 | 3,413 | 2,800 | 0 |
| thanhnien | seen | 3,500 | 7 | 3,500 | 3,460 | 40 | 4 | 3,456 | 3,448 | 2,800 | 0 |
| tuoitre | seen | 3,500 | 6 | 3,500 | 3,500 | 0 | 6 | 3,494 | 3,491 | 2,800 | 0 |
| dantri | seen | 3,501 | 100 | 3,500 | 3,498 | 2 | 7 | 3,491 | 3,466 | 2,800 | 0 |
| saostar | unseen | 500 | 4 | 500 | 500 | 0 | 0 | 500 | 500 | 300 | 0 |
| yan | unseen | 2,181 | 0 | 500 | 500 | 0 | 0 | 500 | 217 | 300 | 0 |
| vietnamnet | unseen | 2,160 | 21 | 500 | 500 | 0 | 0 | 500 | 494 | 300 | 0 |
| nhandan | unseen | 500 | 0 | 500 | 491 | 9 | 0 | 491 | 468 | 300 | 0 |

Tổng: tải được **33,562**, đã chọn 30,000, hợp lệ **29,777**, sau bỏ trùng **29,434**, có sapo **29,068**.

Tải được = `record_id` khác nhau có `extraction_status` success/partial **và có title** trên mọi lượt thử của run crawl (gồm bài ngoài khoảng ngày/không được chọn). *Không title = dòng success/partial không có title (trang tag/listing, ví dụ 2sao `Trang chủ` not_article_page; missing_title), không tính là bài tải được.

### Bài bị loại

| Nguồn | Lỗi | Số bài |
| --- | --- | ---: |
| eva | missing_sapo_policy | 159 |
| cafef | non_headline_title:lottery_code_date | 8 |
| cafef | non_headline_title:lottery_result_today_date | 3 |
| cafef | non_headline_title:lottery_vietlott_date | 2 |
| dantri | non_headline_title:correction_apology_notice | 2 |
| nhandan | non_headline_title:nhandan_generic_title | 5 |
| nhandan | non_headline_title:nhandan_notice_generic_title | 1 |
| nhandan | non_headline_title:nhandan_print_edition_index | 3 |
| thanhnien | non_headline_title:lottery_region_day_result | 15 |
| thanhnien | non_headline_title:lottery_vietlott_result | 7 |
| thanhnien | non_headline_title:thanhnien_print_edition_digest | 18 |

Issue theo loại và quyết định (`data/interim/issues.csv`): `category_unmapped:report` 4, `missing_sapo:report` 370, `missing_sapo_policy:reject` 159, `mojibake_suspected:flag` 2, `non_headline_title:reject` 64.
Sửa kỹ thuật theo trường (`data/interim/fixes.csv`): `sapo` 51, `category_raw` 460, `title` 30.

### Bỏ trùng

- 29,434 dup_group; 341 nhóm ≥2 bài (335 nhóm xuyên nguồn), lớn nhất 3 bài.
- Liên kết tự động: 344 cặp (exact_norm_title: 294, cross_source_jc3: 50).
- Loại khỏi sau bỏ trùng: 343 bài, trong đó `duplicate_of_unseen` 23, `duplicate_in_group` 320.
- Bị chặn: 2 liên kết (separate_decision: 2); `transitive_conflict` = liên kết sẽ gộp hai bài lệch số (hoặc cặp `separate`) qua bài thứ ba.
- Cần review (`dedup_review.csv`): 209 cặp (auto_link_check: 29, date_gap: 31, exact_date_gap: 5, number_conflict: 30, review_band: 114). `auto_link_check`/`exact_date_gap` là liên kết đã áp dụng, cần kiểm tra; độ chính xác liên kết tự động chưa được đo.
- Chuỗi template cùng nguồn: 521 cặp trong 91 chuỗi (`template_series.csv`, chưa gộp).

### Category

- Bài hợp lệ chưa map: 4; mapping chưa duyệt (proposed/needs_review): 0 (approved: 29,773, unmapped: 4).
- Mapping `configs/crawling/category_mapping.csv`, 2 quyết định dedup thủ công.
- Category chung chưa được nhóm duyệt; bước split phải từ chối mapping chưa `approved`.

### Giới hạn

- Tải được (downloaded) = số `record_id` khác nhau trong `articles.jsonl` của run crawl (mọi lượt thử, lấy từ `reports/crawl_stats.csv`) có `extraction_status` success/partial và title không rỗng; gồm cả bài ngoài khoảng ngày hoặc không được chọn vào quota thô. Dòng success/partial không title (trang listing/tag, ví dụ 3.739 dòng 2sao 'Trang chủ' not_article_page, và các dòng missing_title) được báo riêng ở `downloaded_without_title`.
- Hợp lệ (valid) = bài đã chọn qua checking record-level; sau bỏ trùng (after_dedup) = đại diện nhóm trùng không bị loại; có sapo (with_sapo) đếm trong sau bỏ trùng.
- `<unmapped>` ở cột downloaded của `funnel_by_source_category.csv` chủ yếu là bài tải được nhưng không được chọn, có category gốc chưa có trong mapping (mapping được đề xuất từ các bài đã chọn).
- Dedup chỉ so title (Jaccard char-3gram); độ giống sapo chỉ hỗ trợ review, không bao giờ tự liên kết.
- Độ chính xác liên kết tự động chưa được đo. Liên kết gần trùng cần |chênh published_at| <= near_link_max_days; liên kết gần trùng đã áp dụng có JC3 < review_auto_link_below và liên kết title trùng hệt nằm ngoài cửa sổ ngày được liệt kê trong `dedup_review.csv` (auto_linked=True) để kiểm tra mẫu.
- Liên kết bị từ chối khi nó đưa một cặp lệch số hoặc một cặp `separate` thủ công vào cùng nhóm (transitive_conflict); kết quả phụ thuộc thứ tự liên kết (trùng hệt trước, rồi JC3 giảm dần), thứ tự này cố định.
- Chuỗi template cùng nguồn chỉ được báo cáo, không gộp vào dup_group (chờ nhóm quyết định).
- Dòng mapping category là proposed/needs_review cho tới khi nhóm duyệt; khóa chưa map giữ null và `category_mapping_status` null khi category null.
- Phát hiện mojibake dựa trên mẫu ký tự; chỉ loại title có mojibake được xác nhận bằng round-trip.

## File

- `data/raw/articles_raw.jsonl`: 30,000 bài thô đã chọn, đúng 9 trường của plan.
- `reports/crawl_stats.csv`: thống kê crawl theo nguồn (cột `downloaded_by_category` là JSON `[[category gốc, số bài], ...]` dùng cho funnel theo category).
- `reports/funnel_by_source.csv`, `reports/funnel_by_source_category.csv` (category `<unmapped>` hiển thị riêng).
- `data/processed/articles_clean.parquet`: bài hợp lệ kèm `dup_group`.
- `data/interim/`: `issues.csv`, `rejected.csv`, `fixes.csv`, `dedup_pairs.csv`, `dedup_review.csv`, `template_series.csv`.
