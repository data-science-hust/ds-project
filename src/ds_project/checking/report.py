"""reports/data_report.md: one Vietnamese report, crawl thô (from crawl_stats.csv) + clean."""

from __future__ import annotations

import json
from collections import Counter

POLICIES = [
    "Quota thô: 3.500 bài/báo seen, 500 bài/báo unseen; ưu tiên 01/01/2026–21/10/2026, chỉ lùi tới 01/01/2024 khi thiếu.",
    "Inventory duyệt hết sitemap/feed rồi xáo trộn với seed 2026; không đặt quota theo category.",
    "Chỉ lưu 9 trường của plan; không lưu body/HTML/media. Thiếu sapo/category vẫn giữ (partial), không lấy SEO description.",
    "CafeF: timestamp không offset được coi là UTC+07:00 (nhóm đã duyệt).",
    "VietNamNet: cách ly 13 bản ghi sitemap (11 URL năm 2022) có ký tự điều khiển trong URL; không sửa URL nào.",
    "Bỏ qua sitemap tag (`sitemap-tags-NN.xml`) và URL trang tag `-tagNNN.html` (2sao, VietNamNet).",
    "Dân Trí: định dạng 'Dân trí 360' (video ngắn render bằng JS, không có title/ngày trong HTML) và 5 bài emagazine "
    "có tiêu đề dạng ảnh bị loại có hệ thống (quyết định của nhóm), thể hiện ở missing_title.",
    "Dân Trí: sapo của bài emagazine được tải lại riêng (refetch) với selector mới, chỉ nhận khi title tải lại trùng "
    "title đã lưu. `scripts/export_raw.py` điền sapo này vào `articles_raw.jsonl` (chỉ dòng chưa có sapo) và tính lại "
    "`extraction_status` theo đúng quy tắc của crawler (`success` khi đủ title, sapo, category, published_at; còn lại "
    "`partial`). File crawl gốc trong `data/crawl_work/` không bị sửa; danh sách record_id được điền nằm ở "
    "`sapo_refetch/<refetch_id>/sapo_supplement.jsonl` (status `sapo_recovered`) trong thư mục run.",
    "Dân Trí bị giới hạn ~100 request/3 phút (HTTP 428): delay 2,5 s và chờ ≥120 s khi gặp 428.",
    "YAN không có phần sapo riêng; chỉ nhận đoạn đầu được in đậm toàn bộ là sapo.",
]
CRAWL_LIMITATIONS = [
    "Quota thô không phải quota sạch: xem phần Clean (checking, bỏ trùng, funnel).",
    "Inventory duyệt hết chỉ có nghĩa đã duyệt hết endpoint tìm được, không chứng minh bao phủ mọi bài đã đăng.",
]
LIMITATIONS = [
    "Tải được (downloaded) = số `record_id` khác nhau trong `articles.jsonl` của run crawl (mọi lượt thử, lấy từ "
    "`reports/crawl_stats.csv`) có `extraction_status` success/partial và title không rỗng; gồm cả bài ngoài khoảng "
    "ngày hoặc không được chọn vào quota thô. Dòng success/partial không title (trang listing/tag, ví dụ 3.739 dòng "
    "2sao 'Trang chủ' not_article_page, và các dòng missing_title) được báo riêng ở `downloaded_without_title`.",
    "Hợp lệ (valid) = bài đã chọn qua checking record-level; sau bỏ trùng (after_dedup) = đại diện nhóm trùng không "
    "bị loại; có sapo (with_sapo) đếm trong sau bỏ trùng.",
    "`<unmapped>` ở cột downloaded của `funnel_by_source_category.csv` chủ yếu là bài tải được nhưng không được chọn, "
    "có category gốc chưa có trong mapping (mapping được đề xuất từ các bài đã chọn).",
    "Dedup chỉ so title (Jaccard char-3gram); độ giống sapo chỉ hỗ trợ review, không bao giờ tự liên kết.",
    "Độ chính xác liên kết tự động chưa được đo. Liên kết gần trùng cần |chênh published_at| <= near_link_max_days; "
    "liên kết gần trùng đã áp dụng có JC3 < review_auto_link_below và liên kết title trùng hệt nằm ngoài cửa sổ ngày "
    "được liệt kê trong `dedup_review.csv` (auto_linked=True) để kiểm tra mẫu.",
    "Liên kết bị từ chối khi nó đưa một cặp lệch số hoặc một cặp `separate` thủ công vào cùng nhóm "
    "(transitive_conflict); kết quả phụ thuộc thứ tự liên kết (trùng hệt trước, rồi JC3 giảm dần), thứ tự này cố định.",
    "Chuỗi template cùng nguồn chỉ được báo cáo, không gộp vào dup_group (chờ nhóm quyết định).",
    "Dòng mapping category là proposed/needs_review cho tới khi nhóm duyệt; khóa chưa map giữ null và "
    "`category_mapping_status` null khi category null.",
    "Phát hiện mojibake dựa trên mẫu ký tự; chỉ loại title có mojibake được xác nhận bằng round-trip.",
]


def _n(value):
    return f"{int(value):,}"


def crawl_section(crawl_rows):
    total = Counter()
    years = Counter()
    for r in crawl_rows:
        for key in ("selected", "target", "shortfall", "attempted_urls", "downloaded", "downloaded_without_title",
                    "success", "partial", "missing_sapo_before_refetch", "sapo_refetched", "missing_sapo",
                    "missing_category"):
            total[key] += int(r[key])
        years.update({y: int(n) for y, n in json.loads(r["selected_by_year"]).items()})
    statuses = [k for k in crawl_rows[0] if k.startswith("selection_") and any(int(r[k]) for r in crawl_rows)]
    lines = ["## 1. Crawl thô", "",
             "Nguồn: `reports/crawl_stats.csv` (do `scripts/export_raw.py` ghi cùng `data/raw/articles_raw.jsonl`).", "",
             f"- Đã chọn vào quota thô: **{total['selected']:,}/{total['target']:,}**; thiếu **{total['shortfall']:,}**.",
             f"- Năm đăng: {', '.join(f'{y}: {n:,}' for y, n in sorted(years.items()))}.",
             f"- Lượt thử URL bài {total['attempted_urls']:,}; tải được (có title) {total['downloaded']:,}; "
             f"success/partial không title {total['downloaded_without_title']:,}.",
             f"- `extraction_status` trong raw: `success` {total['success']:,}, `partial` {total['partial']:,}.",
             f"- Thiếu sapo sau crawl {total['missing_sapo_before_refetch']:,}; lấy lại bằng refetch "
             f"{total['sapo_refetched']:,}; còn thiếu trong raw **{total['missing_sapo']:,}** "
             f"({100 * total['missing_sapo'] / max(total['selected'], 1):.2f}%). Thiếu category {total['missing_category']:,}.",
             "", "| Nguồn | Nhóm | Run | Inventory URL | Lượt thử | Tải được | Không title | Đã chọn | Target | Thiếu | "
             "Thiếu sapo (crawl) | Refetch | Thiếu sapo (raw) | Thiếu category | Category gốc | Ngày đăng |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in crawl_rows:
        lines.append(f"| {r['source']} | {r['outlet_group']} | `{r['run_id']}` | {_n(r['inventory_urls'] or 0)} | "
                     f"{_n(r['attempted_urls'])} | {_n(r['downloaded'])} | {_n(r['downloaded_without_title'])} | "
                     f"{_n(r['selected'])} | {_n(r['target'])} | {_n(r['shortfall'])} | "
                     f"{_n(r['missing_sapo_before_refetch'])} | {_n(r['sapo_refetched'])} | {_n(r['missing_sapo'])} | "
                     f"{_n(r['missing_category'])} | {r['raw_categories']} | {r['date_min'][:10]} → {r['date_max'][:10]} |")
    lines += ["", "Lý do chọn/không chọn trên mọi lượt thử (`selection_status` của crawler):", "",
              "| Nguồn | " + " | ".join(s.removeprefix("selection_") for s in statuses) + " |",
              "| --- |" + " ---: |" * len(statuses)]
    lines += [f"| {r['source']} | " + " | ".join(_n(r[s]) for s in statuses) + " |" for r in crawl_rows]
    lines += ["", "### Chính sách và quyết định", ""] + [f"- {p}" for p in POLICIES]
    lines += ["", "### Giới hạn", ""] + [f"- {item}" for item in CRAWL_LIMITATIONS]
    return lines


def clean_section(summary, by_source, rejection_counts):
    lines = ["## 2. Clean", "",
             "Input: `data/raw/articles_raw.jsonl`; cột `source_run_id` của parquet lấy theo `run_id` trong "
             "`reports/crawl_stats.csv`.", "",
             "### Funnel: tải được → hợp lệ → sau bỏ trùng → có sapo", "",
             "| Nguồn | Nhóm | Tải được | Không title* | Đã chọn | Hợp lệ | Loại | Loại do trùng | Sau bỏ trùng | Có sapo | Mục tiêu sạch | Thiếu |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in by_source:
        lines.append(f"| {r['source']} | {r['outlet_group']} | {r['downloaded']:,} | {r['downloaded_without_title']:,} | "
                     f"{r['selected']:,} | {r['valid']:,} | "
                     f"{r['rejected']:,} | {r['dedup_excluded']:,} | {r['after_dedup']:,} | {r['with_sapo']:,} | "
                     f"{r['min_clean_articles']:,} | {r['shortfall']:,} |")
    t = summary["totals"]
    lines += ["", f"Tổng: tải được **{t['downloaded']:,}**, đã chọn {t['selected']:,}, hợp lệ **{t['valid']:,}**, "
              f"sau bỏ trùng **{t['after_dedup']:,}**, có sapo **{t['with_sapo']:,}**.", "",
              "Tải được = `record_id` khác nhau có `extraction_status` success/partial **và có title** trên mọi lượt "
              "thử của run crawl (gồm bài ngoài khoảng ngày/không được chọn). *Không title = dòng success/partial không "
              "có title (trang tag/listing, ví dụ 2sao `Trang chủ` not_article_page; missing_title), không tính là bài "
              "tải được.", "", "### Bài bị loại", ""]
    if rejection_counts:
        lines += ["| Nguồn | Lỗi | Số bài |", "| --- | --- | ---: |"]
        lines += [f"| {r['source']} | {r['issue']} | {r['records']} |" for r in rejection_counts]
    else:
        lines.append("Không có bài bị loại.")
    lines += ["", "Issue theo loại và quyết định (`data/interim/issues.csv`): "
              + ", ".join(f"`{k}` {v:,}" for k, v in summary["issue_counts"].items()) + ".",
              "Sửa kỹ thuật theo trường (`data/interim/fixes.csv`): "
              + (", ".join(f"`{k}` {v:,}" for k, v in summary["fix_counts"].items()) or "không có") + "."]
    d = summary["dedup"]
    lines += ["", "### Bỏ trùng", "",
              f"- {d['groups']:,} dup_group; {d['multi_member_groups']:,} nhóm ≥2 bài "
              f"({d['cross_source_groups']:,} nhóm xuyên nguồn), lớn nhất {d['largest_group']} bài.",
              f"- Liên kết tự động: {d['auto_links']:,} cặp ({', '.join(f'{k}: {v}' for k, v in d['auto_link_rules'].items())}).",
              f"- Loại khỏi sau bỏ trùng: {d['excluded']:,} bài, trong đó `duplicate_of_unseen` "
              f"{d['excluded_duplicate_of_unseen']:,}, `duplicate_in_group` {d['excluded_duplicate_in_group']:,}.",
              f"- Bị chặn: {d['blocked_links']:,} liên kết ({', '.join(f'{k}: {v}' for k, v in d['blocked_reasons'].items()) or 'không'}); "
              "`transitive_conflict` = liên kết sẽ gộp hai bài lệch số (hoặc cặp `separate`) qua bài thứ ba.",
              f"- Cần review (`dedup_review.csv`): {d['review_pairs']:,} cặp "
              f"({', '.join(f'{k}: {v}' for k, v in d['review_rules'].items())}). `auto_link_check`/`exact_date_gap` "
              "là liên kết đã áp dụng, cần kiểm tra; độ chính xác liên kết tự động chưa được đo.",
              f"- Chuỗi template cùng nguồn: {d['template_pairs']:,} cặp trong {d['template_series']:,} chuỗi "
              "(`template_series.csv`, chưa gộp)."]
    lines += [f"- Ghi chú: {note}" for note in d["notes"]]
    c = summary["category"]
    lines += ["", "### Category", "",
              f"- Bài hợp lệ chưa map: {c['valid_unmapped']:,}; mapping chưa duyệt (proposed/needs_review): "
              f"{c['valid_unapproved_mapping']:,} ({', '.join(f'{k}: {v:,}' for k, v in c['mapping_status_counts'].items())}).",
              f"- Mapping `{summary['configs']['mapping']}`, {summary['configs']['dedup_decisions']} quyết định dedup thủ công.",
              "- Category chung chưa được nhóm duyệt; bước split phải từ chối mapping chưa `approved`.",
              "", "### Giới hạn", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    return lines


def render(created_at, crawl_rows, summary, by_source, rejection_counts):
    lines = ["# Báo cáo dữ liệu", "", f"Tạo lúc {created_at} bởi `scripts/build_clean.py`.", ""]
    lines += crawl_section(crawl_rows) + [""] + clean_section(summary, by_source, rejection_counts)
    lines += ["", "## File", "",
              f"- `data/raw/articles_raw.jsonl`: {sum(int(r['selected']) for r in crawl_rows):,} bài thô đã chọn, "
              "đúng 9 trường của plan.",
              "- `reports/crawl_stats.csv`: thống kê crawl theo nguồn (cột `downloaded_by_category` là JSON "
              "`[[category gốc, số bài], ...]` dùng cho funnel theo category).",
              "- `reports/funnel_by_source.csv`, `reports/funnel_by_source_category.csv` (category `<unmapped>` hiển thị riêng).",
              "- `data/processed/articles_clean.parquet`: bài hợp lệ kèm `dup_group`.",
              "- `data/interim/`: `issues.csv`, `rejected.csv`, `fixes.csv`, `dedup_pairs.csv`, `dedup_review.csv`, "
              "`template_series.csv`.", ""]
    return "\n".join(lines)
