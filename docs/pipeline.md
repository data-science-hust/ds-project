# Pipeline: crawl → xuất raw → clean → split

Hướng dẫn chạy từng bước và quy tắc của từng bước. Schema các file ở
[data_contract.md](data_contract.md).

## Bố cục dữ liệu

| Vị trí | Nội dung |
| --- | --- |
| `data/crawl_work/<source>/<run_id>/` | Thư mục làm việc của crawler (tạm, không commit): mọi lượt thử, inventory, cache, journal để resume. **Hiện đã nén** vào archive ngoài repo (xem dưới). |
| `data/crawl_work/_batches/<run_id>/` | `status.json` và `<source>.console.log` của batch. |
| `../ds-project-archive/` (ngoài repo, chỉ trên máy người crawl) | `crawl_work_final_runs_20261023.zip` (12 run cuối + tải lại sapo, cần để chạy lại `export_raw.py`) và `crawl_archive_20261023.zip` (pilot, run cũ, log và báo cáo cũ; chỉ để truy vết). |
| `data/raw/articles_raw.jsonl` | **File raw duy nhất**: 30.000 bài đã chọn, đúng 9 trường của plan. |
| `data/interim/` | 6 file log của bước clean (issues, rejected, fixes, dedup). |
| `data/processed/articles_clean.parquet` | Dữ liệu sạch. |
| `data/manifests/` | `split_manifest.csv`, `reserve_list.csv`, `annotation_map.csv` (nội bộ) của bước split. |
| `data/annotations/blind/` | Bảng blind `annotation_id,title,category` gửi annotator. |
| `reports/` | `crawl_stats.csv`, `funnel_by_source.csv`, `funnel_by_source_category.csv`, `data_report.md`, `split_report.md`, `split_counts.csv`. |

Luồng: `crawl_data.py`/`run_crawling_batch.py` → (`refetch_missing_sapo.py`) →
`export_raw.py` → `build_clean.py` → `build_split.py`. Các bước sau chỉ đọc output của bước trước.

Dữ liệu stage 1 đã chốt: chỉ cần chạy lại `build_clean.py`/`build_split.py` khi
đổi config (mapping, rule, quyết định dedup). Muốn crawl thêm hoặc chạy lại
`export_raw.py` thì giải nén run cuối trước (zip lưu đường dẫn tương đối từ root):

```powershell
Expand-Archive ../ds-project-archive/crawl_work_final_runs_20261023.zip -DestinationPath .
```

## Cấu hình đã xác nhận

- 3.500 bài thô/báo seen, 500/báo unseen: tổng 30.000 bài.
- Ưu tiên 01/01/2026–hết 21/10/2026 theo giờ Việt Nam. Hết bài phù hợp mới
  bù lần lượt từ 2025 rồi 2024, không lùi quá 01/01/2024. Đủ quota thì dừng.
  Nếu vẫn thiếu thì báo để nhóm quyết định. Báo cáo ghi số bài chọn theo năm.
- Lập inventory toàn nguồn, loại URL lặp, xáo trộn với seed cố định rồi fetch.
  Không chia quota category; tỷ lệ tự nhiên chỉ được ước lượng trên inventory
  tìm được. Mapping category đã được nhóm duyệt; stratify split theo category
  đã triển khai ở `scripts/build_split.py` (mục Split).
- Chấp nhận mọi định dạng bài khi trích xuất được metadata. Chỉ lưu chín
  trường trong plan, không lưu body/HTML bài/media.
- Riêng CafeF, nhóm đã cho phép coi timestamp không có offset là UTC+07:00.
  Parser ghi quyết định này ở journal. Các nguồn khác không tự áp dụng quy tắc
  đó; timestamp thiếu múi giờ vẫn phải báo kiểm tra.

## Cài đặt và pilot

Python 3.10 trở lên, từ repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/crawl_data.py --sources yan saostar --run-id pilot_metadata_v1 --pilot-per-source 5 --max-discovery-pages 3 --max-attempts 30
```

Pilot cho phép inventory chưa duyệt hết để thử HTTP/parser, không đại diện phân
bố category/thời gian. `--max-attempts` giới hạn tổng số URL bài được thử trong
mỗi run/nguồn qua cả các lần resume, không gồm request discovery và retry.

## Crawl thật và resume

Chạy cả 12 nguồn, hai nguồn độc lập cùng lúc:

```powershell
python scripts/run_crawling_batch.py --run-id metadata_2024_20261021 --workers 2 --reuse-discovery-cache
```

Mỗi nguồn vẫn giữ request delay riêng. Batch ưu tiên YAN và SAOstar trước.
Trạng thái/PID/quota: `data/crawl_work/_batches/<run_id>/status.json`; stdout
của từng nguồn: `<source>.console.log` cùng thư mục. Trong lúc fetch,
`data/crawl_work/<source>/<run_id>/journal/progress.json` cập nhật sau mỗi 25
bài. Nguồn kết thúc với exit code 0 là `complete`, còn lại `needs_review`.
Cache reuse chỉ chép XML/metadata discovery của các run trước trong
`data/crawl_work/<source>/`, có ghi provenance ở `cache_reuse.json`; không trộn
bài/selection giữa các run. Snapshot config thay đổi cần run ID mới.

Yêu cầu batch đang chạy dừng:

```powershell
python scripts/run_crawling_batch.py --run-id metadata_2024_20261021 --stop
```

Nếu muốn chạy từng bước thủ công:

```powershell
python scripts/crawl_data.py --sources all --stage discover --run-id metadata_20261021
python scripts/crawl_data.py --sources all --stage fetch --run-id metadata_20261021
```

Hoặc `python scripts/crawl_data.py --sources all --run-id metadata_20261021`
để chạy discovery → fetch lần lượt từng nguồn. Chạy lại cùng lệnh/run ID để
resume. Script đối chiếu snapshot config/profile; thay đổi cấu hình cần run ID
mới. Không tự retry lại các bài đã ghi lỗi; sau sửa parser/network, dùng run
mới để giữ kết quả gốc.

Mặc định serial, cách request ít nhất một giây, có timeout và retry giới hạn cho
lỗi mạng/429/5xx. Không retry 403 hàng loạt. Cấu hình tại
`configs/crawling/run.json`; danh sách nguồn/quota clean từ plan tại `sources.csv`.

Nếu chạm `--max-discovery-pages` (mặc định 2.000/nguồn) hoặc có endpoint lỗi,
production fetch dừng để tránh lấy mẫu từ inventory dở. Đọc discovery report,
sửa lỗi/tăng budget rồi discovery lại; cache tránh tải lại phần đã tìm.
Mỗi run có lock `.running`; sau kill cứng chỉ xóa lock khi đã xác minh process
đó dừng. Không chạy hai process ghi cùng source/run ID.

SAOstar dùng sitemap tháng trực tiếp vì index năm 2026 chuyển hướng về home.
YAN dùng endpoint public `ContentList/TinMoiLoadMore` từ JavaScript của website,
duyệt Tin mới và toàn bộ category được menu liên kết; Tin mới riêng có thể quá
ít URL. Profile YAN có User-Agent tương thích trình duyệt; không cần tài khoản.

### Thư mục làm việc của một run

Trong `data/crawl_work/<source>/<run_id>/`:

- `articles.jsonl`: đúng chín trường, gồm cả lượt fetch/parse lỗi.
- `discovered_urls.jsonl`, `discovery_cache/`, `discovery_report.json`: inventory,
  xuất xứ URL, cache XML/metadata đã parse và giới hạn coverage. Không cache HTML.
- `run_manifest.json`: snapshot cấu hình để resume; `cache_reuse.json` nếu dùng lại cache.
- `selected_record_ids.csv`: ID tính vào target thô, thuộc primary hay fallback.
- `crawl_report.json`: số bài phù hợp, thiếu quota và lý do chưa chọn.
- `journal/`: `requests.jsonl`, `article_attempts.jsonl` (resume, selector
  evidence, URL cuối sau redirect, lỗi), `robots.txt`, `progress.json`, và
  `failure.json` nếu nguồn chưa hoàn thành.
- `sapo_refetch/<refetch_id>/`: kết quả tải lại sapo (xem dưới).

ID là SHA-256 của source + URL discovery đã bỏ fragment; không tự xóa query.
`url` giữ URL request; redirect cuối ghi ở journal. Bài tính quota cần title,
metadata xác nhận là trang bài báo, ngày đăng xác định trong khoảng cho phép,
không có ký tự replacement trong title/sapo/category. Thiếu sapo/category vẫn
giữ và báo `partial`, không tự lấy SEO description hoặc body để giả làm sapo.

Inventory duyệt hết chỉ có nghĩa đã duyệt hết endpoint tìm được, **không chứng
minh bao phủ toàn bộ lịch sử báo**. Mọi shortfall/endpoint lỗi được báo rõ.
Quota thô không chứng minh đủ quota sạch sau dedup.

VietNamNet/Nhân Dân bật phục hồi XML lỗi sau khi đối chiếu crawler cũ. Parser
phục hồi phải kiểm tra đủ bản ghi và đối chiếu từng `loc` từ nguồn bằng parser
strict, không dùng URL đã bị recovering parser tự sửa. `xml_recoveries` trong
discovery report ghi chi tiết. XML bị cắt mất phần cuối, mất cấu trúc hoặc URL
không xác minh được vẫn làm inventory incomplete. Chế độ cách ly URL hỏng là
tùy chọn riêng `quarantine_invalid_sitemap_urls`, mặc định tắt. Ngày 23/10/2026
nhóm cho bật riêng VietNamNet: cách ly 13 bản ghi (11 URL năm 2022) có ký tự
điều khiển trong `loc`, không sửa URL; danh sách ở
`xml_recoveries[].quarantined_urls` của run `metadata_fix_20261023`.

Discovery bỏ qua sitemap `sitemap-tags-NN.xml` và URL trang tag `-tagNNN.html`
(2sao, VietNamNet): trang tag thỏa mẫu ID bài nhưng không phải bài báo.

Dân Trí giới hạn khoảng 100 request/3 phút và trả HTTP 428. Profile đặt
`request_delay_seconds` 2,5 và `rate_limit_cooldown_seconds` 120 (chờ trước khi
retry 428; `Retry-After` dài hơn được ưu tiên). Hai định dạng Dân Trí bị loại có
hệ thống theo quyết định nhóm: "Dân trí 360" (video ngắn render bằng JavaScript,
HTML không có title/ngày) và emagazine có tiêu đề là ảnh; cả hai hiện là
`missing_title`. YAN không có phần sapo riêng: chỉ nhận đoạn đầu `#contentBody`
in đậm toàn bộ (`sapo_require_full_bold_paragraph`).

### Tải lại sapo cho bài đã chọn

Sau khi sửa selector sapo, không cần crawl lại cả nguồn: tải lại đúng các bài
đã chọn đang thiếu sapo, ghi file bổ sung cạnh run, `articles.jsonl` của run
không đổi. Sapo chỉ được nhận khi title tải lại trùng title đã lưu. Chạy lại
cùng lệnh để resume; script tự dừng sau nhiều lỗi liên tiếp (ví dụ 428).

```powershell
python scripts/refetch_missing_sapo.py --source dantri --run-id metadata_2024_20261021 --refetch-id sapo_refetch_dantri_20261023 --request-delay 4
```

Output: `data/crawl_work/<source>/<run_id>/sapo_refetch/<refetch_id>/` gồm
`sapo_supplement.jsonl` (`sapo_recovered`/`sapo_absent`/`title_changed`),
`manifest.json`, `report.json`, `journal/` (request log). Khai báo ở
`sapo_supplements` trong `configs/crawling/final_runs.json` để `export_raw.py`
điền vào raw.

## Xuất raw

`configs/crawling/final_runs.json` ghi run cuối cùng của mỗi nguồn (`runs`) và
lần tải lại sapo (`sapo_supplements`).

```powershell
python scripts/export_raw.py            # lần đầu
python scripts/export_raw.py --replace  # ghi đè sau khi đổi final_runs.json
```

Script chỉ đọc `data/crawl_work/` và ghi:

- `data/raw/articles_raw.jsonl`: UTF-8, mỗi dòng một object với đúng 9 trường
  (`record_id, source, url, title, sapo, category, published_at, crawled_at,
  extraction_status`). Thứ tự: nguồn theo `sources.csv`, trong mỗi nguồn theo
  `selected_record_ids.csv`.
- `reports/crawl_stats.csv`: mỗi nguồn một dòng: run, inventory, lượt thử,
  `downloaded`/`downloaded_without_title`, đã chọn/target/thiếu, số lượt theo
  từng `selection_status` (`selection_*`), năm đăng, `success`/`partial`,
  `missing_<trường>` trên raw xuất ra, `missing_sapo_before_refetch`,
  `sapo_refetched`, số category gốc, khoảng ngày, và `downloaded_by_category`
  (JSON `[[category gốc, số bài], ...]`, dùng cho funnel theo category).

**Sapo tải lại:** chỉ điền vào dòng chưa có sapo, chỉ dùng dòng
`sapo_recovered`. Sau khi điền, `extraction_status` của dòng đó được tính lại
đúng quy tắc của crawler: `success` khi có đủ title, sapo, category,
published_at; còn lại `partial`. Vì vậy 310 bài Dân Trí được lấy lại sapo có
status `success` trong raw. Raw không có cờ riêng cho các bài này; danh sách
record_id nằm ở `sapo_supplement.jsonl` (status `sapo_recovered`) của lần tải
lại (nằm trong `crawl_work_final_runs_20261023.zip`).

Script từ chối (không ghi gì) khi một run: không tồn tại, còn lock `.running`,
thiếu `selected_record_ids.csv`/`crawl_report.json`/`discovery_report.json`,
`shortfall > 0`, là run pilot (`--pilot-per-source`), chọn ít hơn
`raw_targets` của nhóm nguồn trong `configs/crawling/run.json`, có ID được chọn mà không có dòng trong `articles.jsonl` (hoặc
có nhiều dòng), JSONL lỗi; hoặc khi lần tải lại sapo khai báo chưa có
`report.json` hay `remaining > 0`. Output đã có thì cần `--replace`.

## Clean: checking, category mapping, dedup

```powershell
python scripts/build_clean.py            # lần đầu
python scripts/build_clean.py --replace  # ghi đè output cũ
```

Input: `data/raw/articles_raw.jsonl` và `reports/crawl_stats.csv` (thiếu thì
báo lỗi, chạy `export_raw.py` trước). Số bài mỗi nguồn trong raw phải khớp cột
`selected` của `crawl_stats.csv`. Output (đã tồn tại thì cần `--replace`):

| Vị trí | File |
| --- | --- |
| `data/interim/` | `issues.csv` (record_id, lỗi, quyết định reject/flag/report, lý do), `rejected.csv`, `fixes.csv` (trước/sau), `dedup_pairs.csv`, `dedup_review.csv`, `template_series.csv` |
| `data/processed/` | `articles_clean.parquet` (`source_run_id` lấy theo `run_id` của `crawl_stats.csv`) |
| `reports/` | `funnel_by_source.csv`, `funnel_by_source_category.csv`, `data_report.md` |

`data_report.md` là báo cáo duy nhất, hai phần: **Crawl thô** (bảng từ
`crawl_stats.csv`, lý do chọn/không chọn, chính sách đã áp dụng) và **Clean**
(funnel, bài bị loại, số issue/sửa kỹ thuật, bỏ trùng, category, giới hạn).

Định nghĩa funnel: **tải được** = số `record_id` khác nhau trong toàn bộ
`articles.jsonl` của run crawl có `extraction_status` `success`/`partial` **và
có title** (gồm cả bài ngoài khoảng ngày/không được chọn); dòng không có title
(trang tag/listing, missing_title) báo riêng ở `downloaded_without_title`;
**hợp lệ** = bài được chọn qua checking; **sau bỏ trùng** = đại diện nhóm
không bị loại; **có sapo** tính trong sau bỏ trùng. `shortfall` so với
`min_clean_articles` của `sources.csv`.

Cấu hình trong `configs/checking/`: `rejection_rules.json` (rule loại title
không phải tiêu đề bài: Nhân Dân "Tin tức"/trang báo in, Thanh Niên tin báo in,
kết quả xổ số/Vietlott, thông báo cải chính; mọi lượt khớp được log),
`dedup.json` (ngưỡng JC3, kiểu chặn số, `near_link_max_days` = 7: cặp gần trùng
cách nhau hơn 7 ngày không tự gộp mà vào review `date_gap`),
`dedup_decisions.csv` (`link`/`separate` sau review). Liên kết bị từ chối
(`transitive_conflict`) nếu nó gộp một cặp lệch số hoặc cặp `separate` vào cùng
nhóm qua bài thứ ba.

Sau khi đổi run cuối, thêm đề xuất mapping cho category mới (không ghi đè dòng
cũ), rồi review thủ công trước khi chạy clean. `--all-rows` dùng category của
mọi bài tải được (từ `crawl_stats.csv`) thay vì chỉ bài trong raw:

```powershell
python scripts/propose_category_mapping.py --dry-run
python scripts/propose_category_mapping.py
```

## Split: manifest, reserve, bảng blind

```powershell
python scripts/build_split.py            # lần đầu
python scripts/build_split.py --replace  # ghi đè output cũ
```

Input (chỉ đọc): `data/processed/articles_clean.parquet`,
`data/interim/{template_series,dedup_review,dedup_pairs}.csv` (thiếu thì chạy
`build_clean.py` trước), `configs/checking/dedup_decisions.csv`, quota ở
`configs/crawling/sources.csv`, seed ở `configs/crawling/run.json` (2026).
Output (đã tồn tại thì cần `--replace`):

| Vị trí | File |
| --- | --- |
| `data/processed/` | `dataset.csv`: file tổng hợp 23.600 bài đã chia (`record_id, annotation_id, split, is_human_subset, source, outlet_group, category, title, sapo, published_at, url, dup_group`; `annotation_id` có ở 10.000 bài human) |
| `data/manifests/` | `split_manifest.csv` (23.600 dòng), `reserve_list.csv`, `annotation_map.csv` (10.000 dòng, nội bộ) |
| `data/annotations/blind/` | `blind_dev.csv` (1.400 dòng, gán nhãn trước), `blind_train_subset_test.csv` (8.600 dòng); chỉ `annotation_id,title,category` |
| `reports/` | `split_report.md`, `split_counts.csv` (source, category, split, records) |

Quy tắc:

- Pool = đại diện nhóm trùng không có `dedup_excluded_reason`. Script từ chối
  chạy nếu một bài trong pool có category khác null mà mapping chưa
  `approved`. Bài category null không được chọn, nằm trong reserve với stratum
  `(không có category)`.
- Đơn vị lấy mẫu là `split_group`: các `dup_group` nối bởi một cặp gần trùng
  mà clean liệt kê nhưng không gộp (`template_series.csv`: chuỗi tiêu đề theo
  khuôn cùng nguồn; `dedup_review.csv`: cặp chờ review; `dedup_pairs.csv`:
  liên kết bị chặn) được gộp để hai tiêu đề gần giống hệt không nằm ở hai
  split; các bài được giữ, không xóa. Cặp seen–unseen không thể chung split:
  nếu chưa có quyết định `separate` trong `dedup_decisions.csv`, bài seen
  (cả `split_group` của nó) không được chọn, nằm trong reserve với
  `hold_reason=pending_dedup_review_with_unseen`. Nên review các cặp này trước
  khi gán nhãn: đổi quyết định rồi chạy lại clean + split sẽ đổi split. Quota
  đếm theo bài: nhóm không vừa phần quota còn lại thì bỏ qua, xét nhóm kế tiếp;
  ô nào không lấp đúng quota thì script dừng và báo số liệu.
- Mỗi nguồn seen: train 2.500 (trong đó subset human 800) + dev 175 + test_seen
  125; mỗi nguồn unseen: test_unseen 300. Quota từng split (cả subset và
  test_unseen) chia theo tỷ lệ category của pool trong nguồn bằng phần dư lớn
  nhất (hòa thì theo tên category). Không dùng nhãn.
- Dev chỉ lấy bài có sapo (mục 3.1). Train/subset/test không yêu cầu sapo.
- Thứ tự: mỗi (nguồn, category) xáo các `split_group` bằng
  `random.Random("<seed>:<source>:<category>")`; lấp dev → test_seen → train
  (unseen: test_unseen), rồi chọn subset trong train với seed riêng
  (`...:human_subset`). Nhóm trải nhiều ô được xét tại vị trí tương đối sớm
  nhất của nó, mọi ô tiến cùng nhịp. Bài còn lại vào reserve, `reserve_rank`
  tiếp nối thứ tự seed trong (nguồn, category); dùng để thay bài bị loại khi
  gán nhãn (mục 2.4): lấy `reserve_rank` nhỏ nhất cùng nguồn/category có
  `hold_reason` rỗng, lấy cả `split_group`; thay bài dev thì bài thay thế phải
  có `has_sapo=True`.
- `annotation_id` = `A` + 5 chữ số lấy ngẫu nhiên có seed từ 10000–99999; dòng
  của bảng blind theo một hoán vị seed, trộn nguồn/category/split.
- Kiểm tra (dừng nếu sai): đúng quota từng nguồn × split và subset; mỗi
  `split_group`/`dup_group` ở đúng một split và không tách sang reserve; không
  bài nào trong train/dev/test_seen chung nhóm với bài unseen; không cặp gần
  trùng chưa gộp nào (trừ cặp `separate`) nằm ở hai split; bài bị giữ chờ
  review không vào split; subset ⊂ train; dev đủ sapo; bảng
  blind/ánh xạ phủ đúng 10.000 bài; build hai lần trong cùng lệnh cho output
  giống hệt từng byte. Báo cáo không ghi thời điểm chạy nên chạy lại với cùng
  input cho file giống hệt.

`annotation_assignment.csv` (mục 2.2: chunk C1..C9, người P1..P9) không do
script này tạo; lead human annotation lập từ `annotation_map.csv` và bảng blind.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```
