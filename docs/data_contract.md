# Data contract cho crawling/checking

Chín trường đầu dưới đây theo mục 1.1 của plan. Quy ước ID, định dạng file,
timestamp và các giá trị status là đề xuất triển khai của repo.

Crawler hiện chỉ ghi đúng chín trường của mục 1.1 vào `articles.jsonl`, theo
xác nhận của nhóm. Dataset thô bàn giao là `data/raw/articles_raw.jsonl`
(`scripts/export_raw.py`): đúng chín trường này, theo thứ tự nguồn trong
`sources.csv` rồi thứ tự chọn trong `selected_record_ids.csv`. Ba trường `category_raw`, `title_raw`, `dup_group` phía dưới
là contract cho bước clean/dedup tiếp theo, không được thêm vào output crawl.
Evidence/parser/resume nằm riêng trong journal
(`data/crawl_work/<source>/<run_id>/journal/`); không lưu HTML/body.
Ngoại lệ timestamp đã được nhóm chấp thuận: CafeF không ghi offset thì dùng
UTC+07:00. Quyết định này cấu hình riêng nguồn và được ghi evidence trong log.

| Trường | Kiểu | Quy ước |
| --- | --- | --- |
| `record_id` | string | ID ổn định cho bài, không dùng index của dataframe. Đề xuất hash source + canonical URL; chốt quy tắc canonicalization trước crawl đầy đủ. |
| `source` | string | Slug trong `configs/crawling/sources.csv`, không dùng làm nhãn clickbait. |
| `url` | string/null | URL nguồn; bắt buộc không rỗng để vào clean. Giữ URL gốc trong raw; canonical URL có thể lưu thêm riêng. |
| `title` | string/null | Tiêu đề gốc; bắt buộc không rỗng/whitespace để vào clean. Không thay đổi phong cách viết. |
| `sapo` | string/null | Lead đoạn mở đầu; thiếu thì null, không tự lấy toàn body hoặc điền title thay thế. |
| `category` | string/null | Raw: tên category nguồn. Clean: category chung đã mapping. Chưa map phải được báo cáo. |
| `published_at` | string/null | ISO 8601 có offset khi xác định được; không tự bịa ngày hoặc timezone. |
| `crawled_at` | string | ISO 8601 có offset; thời điểm fetch thực tế. |
| `extraction_status` | string | `success` (đủ title, sapo, category, published_at), `partial`, `fetch_error`, `parse_error`; không phải kết luận kiểm tra dữ liệu. Raw xuất ra chỉ có bài `success`/`partial`; dòng được điền sapo tải lại thì status tính lại theo cùng quy tắc. |
| `category_raw` | string/null | Giữ category gốc khi tạo clean để truy vết mapping. |
| `title_raw` | string/null | Bản title trước sửa kỹ thuật trong clean; raw vẫn giữ nguyên. |
| `dup_group` | string | Bắt buộc ở clean trước split; bài đơn lẻ cũng có nhóm riêng. |

## Cột của `articles_clean.parquet`

Mỗi dòng là một bài **hợp lệ** (đã qua checking), kể cả bài không phải đại diện
nhóm trùng để split lấy mẫu theo nhóm. Bài bị loại nằm ở
`data/interim/rejected.csv`, raw giữ nguyên.

| Cột | Kiểu | Quy ước |
| --- | --- | --- |
| `record_id`, `source`, `url` | string | Như raw; `record_id` duy nhất. |
| `outlet_group` | string | `seen`/`unseen` theo `sources.csv`. |
| `title` | string | `title_raw` sau strip khoảng trắng hai đầu + NFC; không đổi chữ/hoa thường/dấu câu. |
| `title_raw` | string | Chuỗi title gốc trong raw. |
| `sapo` | string/null | Strip + NFC; rỗng → null. |
| `category` | string/null | Nhóm chung từ `category_mapping.csv`; null khi chưa map. |
| `category_raw` | string/null | Category nguồn sau `html.unescape` + NFC + strip; rỗng → null. Là khóa join mapping. |
| `category_mapping_status` | string/null | `review_status` của dòng mapping (`proposed`/`needs_review`/`approved`); null khi `category` null (khóa thiếu hoặc dòng mapping chưa điền category). Split chỉ nhận bài có `category` khác null **và** status `approved`. |
| `published_at`, `crawled_at`, `extraction_status` | string | Như raw. |
| `dup_group` | string | `dg_` + 12 ký tự hex đầu của SHA-256 các `record_id` thành viên đã sắp xếp; bài đơn cũng có nhóm. |
| `dup_group_size` | int | Số bài hợp lệ trong nhóm. |
| `is_dup_representative` | bool | Bài được giữ của nhóm. |
| `dedup_excluded_reason` | string/null | `duplicate_in_group` hoặc `duplicate_of_unseen` (bản seen trùng bài unseen, loại khỏi mọi split seen); null khi giữ. |
| `has_sapo` | bool | Có sapo sau sửa kỹ thuật. |
| `source_run_id` | string | Run crawl tạo bài: `configs/crawling/final_runs.json`, ghi qua cột `run_id` của `reports/crawl_stats.csv`. |

Đại diện: nhóm có bài unseen giữ một bản unseen (sớm nhất theo `published_at`,
rồi `record_id`); nhóm chỉ có seen giữ bài sớm nhất, rồi `record_id`.
"Sau bỏ trùng" = `is_dup_representative` và không có `dedup_excluded_reason`.

Raw JSONL dùng UTF-8, mỗi dòng là một object. Trong thư mục làm việc của
crawler, dữ liệu chưa extract được vẫn được lưu kèm status. Log request giữ HTTP status/lỗi và tham chiếu tới bản raw.
Tách biệt fetch thành công với record hợp lệ: `success` không bảo đảm pass checking.

## Checking

- Required fields, kiểu dữ liệu, ID trùng, thiếu URL/title, title chỉ có khoảng trắng.
- URL không hợp lệ hoặc không thuộc nguồn dự kiến, timestamp không parse được.
- Encoding hỏng hoặc nghi ngờ mojibake: đưa vào review, chỉ sửa khi có bằng chứng.
- Trùng URL/canonical URL; trùng title và gần trùng xuyên nguồn. Chuẩn hóa để
  so sánh trên trường phụ, không ghi đè title hiển thị.
- Sapo thiếu; category thiếu/chưa map; thống kê theo source/category.
- Xuất issue log theo `record_id`, loại lỗi, quyết định và lý do. Mẫu không đủ
  điều kiện clean vẫn được giữ trong raw và bảng rejected.
- Trước freeze: mọi clean record có ID duy nhất, title/URL hợp lệ, `dup_group`,
  category được xử lý; quota đáp ứng sau dedup và loại leakage.

Các mục trên đã triển khai trong `scripts/build_clean.py`; rule loại title
không phải tiêu đề bài nằm ở `configs/checking/rejection_rules.json`. Mojibake
chỉ loại khi title round-trip cp1252/latin-1 → UTF-8 xác nhận; còn lại gắn cờ
review. Mọi dòng mapping đã `approved` (16 nhóm); split từ chối chạy nếu còn
dòng chưa duyệt được dùng. Dòng mapping
`approved` bắt buộc có category. Dedup: liên kết gần trùng cần cách nhau ≤ 7
ngày đăng và không được gộp bắc cầu hai bài lệch số; xem `dedup_review.csv`.

## Các bảng bàn giao sau stage checking

Các bảng của bước split do `scripts/build_split.py` sinh (UTF-8 có BOM, CSV
chuẩn; title có xuống dòng được đặt trong ngoặc kép). Chạy lại với cùng input
và seed cho file giống hệt từng byte.

`data/manifests/split_manifest.csv`: sáu cột đầu đúng thứ tự của plan, ba cột
sau là bổ sung của repo; một dòng cho mỗi bài được chọn (23.600 dòng).

```csv
record_id,source,split,dup_group,is_human_subset,seed,split_group,category,outlet_group
```

| Cột | Quy ước |
| --- | --- |
| `split` | `train`, `dev`, `test_seen`, `test_unseen`. |
| `is_human_subset` | `True`/`False`; `True` chỉ cho subset 6.400 **trong train**. Dev/test vẫn thuộc phạm vi human annotation; không hiểu cờ này là toàn bộ phạm vi human. |
| `seed` | Seed của `configs/crawling/run.json` (2026). |
| `split_group` | Đơn vị lấy mẫu: `sg_` + 12 ký tự hex đầu của SHA-256 các `dup_group` thành viên (sắp xếp, nối bằng xuống dòng). Gộp các `dup_group` nối bởi một cặp gần trùng chưa gộp cùng phía seen/unseen (`data/interim/template_series.csv`, `dedup_review.csv`, liên kết bị chặn trong `dedup_pairs.csv`); nhóm không nối thì có một `dup_group`. |
| `category`, `outlet_group` | Như `articles_clean.parquet` (category luôn khác null và `approved`). |

Một `dup_group` hay `split_group` không xuất hiện ở nhiều split và không bị
tách giữa split và reserve.

`data/manifests/reserve_list.csv`: mọi bài sau bỏ trùng không được chọn.

```csv
record_id,source,category,dup_group,split_group,reserve_rank,has_sapo,hold_reason
```

`category` = `(không có category)` cho bài category null (không được chọn).
`reserve_rank` bắt đầu từ 1 trong mỗi (`source`, `category`), tiếp nối thứ tự
seed của bước split. `has_sapo` (`True`/`False`) như bảng clean. `hold_reason`
rỗng = dùng được; `no_category` = nhóm có bài category null;
`pending_dedup_review_with_unseen` = nhóm có bài seen gần trùng một bài unseen
mà chưa có quyết định dedup (không được chọn tới khi review xong). Bài thay
thế: `reserve_rank` nhỏ nhất cùng nguồn/category có `hold_reason` rỗng, lấy cả
`split_group` nếu nhóm có nhiều bài; **thay bài dev thì cả nhóm phải có
`has_sapo=True`** (bỏ qua bài không sapo).

Bảng blind trong `data/annotations/blind/`: `blind_dev.csv` (1.400 bài dev,
gán trước) và `blind_train_subset_test.csv` (8.600 = subset 6.400 + test_seen
1.000 + test_unseen 1.200), chỉ ba cột:

```csv
annotation_id,title,category
```

`annotation_id` = `A` + 5 chữ số (lấy ngẫu nhiên có seed từ 10000–99999), không
suy ra được nguồn, split hay `record_id`; thứ tự dòng là hoán vị có seed.
`title` là title của clean (không sửa thêm); `category` là nhóm chung đã duyệt (16 nhóm).

`data/manifests/annotation_map.csv` (nội bộ, không gửi annotator), 10.000 dòng
sắp theo `annotation_id`:

```csv
annotation_id,record_id,split,source
```

`reports/split_counts.csv`: `source,category,split,records`, với `split` gồm
`train`, `train_human_subset` (con của train), `dev`, `test_seen`,
`test_unseen`, `reserve`.

`category_mapping.csv`:

```csv
source,category_raw,category,review_status,notes
```

Khóa mapping là `(source, category_raw)` ở dạng đã chuẩn hóa (`html.unescape`
+ NFC + strip); một khóa chỉ có một category chung thuộc 16 nhóm trong
`src/ds_project/checking/categories.py`. `review_status` là `proposed`,
`needs_review` hoặc `approved` (chỉ nhóm được đặt); category rỗng = chưa map.
`review_status` và `notes` là metadata bổ sung của repo.

Bảng blind chỉ gồm `annotation_id,title,category`. Bảng ánh xạ annotation ID → record ID
và `annotation_assignment.csv` nằm trong manifests nội bộ, không đính kèm bảng
blind. `annotation_assignment.csv` (mục 2.2: chunk C1..C9, người P1..P9) do
lead human annotation lập; schema chốt khi chia chunk, bước split không tạo.
