# DS Project 2026.1 — Vietnamese Clickbait Detection (Group 5)

Phát hiện tiêu đề clickbait trên báo điện tử tiếng Việt, theo plan
**DS Project 2026.1 Group 5.pdf**. Repo này chứa pipeline dữ liệu của
**stage 1 (mục 1.1–1.2)**: crawl → kiểm tra/làm sạch → bỏ trùng → mapping
category → chia tập và bảng blind cho human annotation.

**Trạng thái:** stage 1 đã xong. Bước tiếp theo là human annotation (mục 2):
lead annotation lập `annotation_assignment.csv` và gán nhãn bảng dev trước.

## Kết quả stage 1

| Bước | Số bài |
| --- | ---: |
| Crawl thô: 3.500/báo seen × 8, 500/báo unseen × 4, toàn bộ đăng năm 2026 | 30.000 |
| Hợp lệ sau kiểm tra (loại 64 tiêu đề không phải tin + 159 bài Eva thiếu sapo) | 29.777 |
| Sau bỏ trùng/gần trùng xuyên nguồn | 29.434 |
| **Chia tập** | **23.600** |
| — train (trong đó subset human 6.400) | 20.000 |
| — dev (100% có sapo) | 1.400 |
| — test seen | 1.000 |
| — test unseen (saostar, yan, vietnamnet, nhandan) | 1.200 |
| Dự phòng để thay bài lỗi khi gán nhãn | 5.834 |
| Phạm vi human annotation (subset + dev + 2 test) | 10.000 |

Nguồn seen: afamily, 2sao, eva, kenh14, cafef, thanhnien, tuoitre, dantri.
Category gốc được map về 16 nhóm chung (đã duyệt); mỗi nguồn seen chia
train/dev/test theo cùng tỷ lệ category. Chi tiết: `reports/data_report.md`,
`reports/split_report.md`, `reports/category_mapping_review.md`.

## Cấu trúc

```text
configs/
  crawling/   sources.csv (nguồn, quota), run.json (khoảng ngày, seed), profiles.json (selector từng báo),
              final_runs.json (run cuối mỗi nguồn), category_mapping.csv (category gốc → nhóm chung)
  checking/   rejection_rules.json, dedup.json, dedup_decisions.csv
src/ds_project/
  crawling/   discovery sitemap/feed, tải trang, trích xuất 9 trường
  checking/   kiểm tra, sửa kỹ thuật, mapping, dedup, báo cáo clean
  splitting/  chia tập, dự phòng, bảng blind, báo cáo split
scripts/      CLI cho từng bước (bảng dưới)
tests/        unittest
docs/         pipeline.md (cách chạy + quy tắc), data_contract.md (schema)
data/         không commit: raw/, interim/, processed/, manifests/, annotations/blind/
reports/      không commit: báo cáo sinh ra từ pipeline
```

## Cài đặt

Python 3.10+:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Chạy pipeline

| Bước | Lệnh | Output chính |
| --- | --- | --- |
| 1. Crawl | `python scripts/run_crawling_batch.py --run-id <id> --workers 2` | `data/crawl_work/<source>/<run_id>/` |
| (tùy chọn) Tải lại sapo | `python scripts/refetch_missing_sapo.py --source <s> --run-id <id> --refetch-id <id>` | `.../sapo_refetch/<id>/` |
| 2. Xuất raw | `python scripts/export_raw.py` | `data/raw/articles_raw.jsonl`, `reports/crawl_stats.csv` |
| 3. Clean + dedup | `python scripts/build_clean.py` | `data/processed/articles_clean.parquet`, `data/interim/*.csv`, `reports/data_report.md` |
| 4. Split | `python scripts/build_split.py` | `data/manifests/*.csv`, `data/annotations/blind/*.csv`, `reports/split_report.md` |

Thêm `--replace` để ghi đè output đã có. Dữ liệu đã chốt, nên thường chỉ chạy
lại bước 3–4 khi đổi config. Thư mục làm việc của crawler và log crawl được
lưu dạng zip ngoài repo (`../ds-project-archive/`, chỉ có trên máy người crawl);
giải nén trước khi chạy lại bước 1–2. Hướng dẫn đầy đủ:
[docs/pipeline.md](docs/pipeline.md).

## Deliverables stage 1

| Plan | File |
| --- | --- |
| Dữ liệu sạch có `dup_group` | `data/processed/articles_clean.parquet` |
| Mapping category | `configs/crawling/category_mapping.csv` |
| Báo cáo số lượng (tải được → hợp lệ → sau bỏ trùng → có sapo) | `reports/data_report.md`, `reports/funnel_by_source*.csv` |
| Split manifest (`record_id, source, split, dup_group, is_human_subset, seed`) | `data/manifests/split_manifest.csv` |
| Bảng blind (`annotation_id, title`) | `data/annotations/blind/blind_dev.csv` (1.400), `blind_train_subset_test.csv` (8.600) |
| Ánh xạ annotation nội bộ (không gửi annotator) | `data/manifests/annotation_map.csv` |
| Danh sách dự phòng | `data/manifests/reserve_list.csv` |

## Quy tắc chính

- Chỉ lưu 9 trường của plan (`record_id, source, url, title, sapo, category,
  published_at, crawled_at, extraction_status`); không lưu body/HTML/media.
- Giữ nguyên cách viết tiêu đề; chỉ sửa kỹ thuật (khoảng trắng, Unicode NFC) và
  ghi log trước/sau.
- Không dùng nhãn (LLM hay người) để chọn mẫu; không ép cân bằng nhãn.
- Nhóm trùng/gần trùng và chuỗi tiêu đề cùng khuôn luôn nằm trong một split;
  bài seen trùng bài unseen bị loại khỏi phía seen.
- Bảng blind chỉ có `annotation_id, title`; không gửi nguồn, URL, split.

## Việc bàn giao tiếp

- **P3 (lead annotation):** lập `annotation_assignment.csv` (9 chunk) từ bảng
  blind, gán nhãn dev trước; thay bài lỗi bằng `reserve_list.csv` theo quy tắc
  trong `docs/pipeline.md`.
- **P2 (LLM):** dùng dev (có sapo, category chung) cho 4 biến thể input; gọi LLM
  từng tiêu đề hoặc trộn ngẫu nhiên batch, ghi rõ category chỉ là ngữ cảnh.
