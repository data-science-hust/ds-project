"""reports/split_report.md and reports/split_counts.csv (Vietnamese report, no timestamp: byte-identical reruns)."""

from __future__ import annotations

from collections import Counter, defaultdict

from .split import HOLD_NO_CATEGORY, HOLD_UNSEEN, NO_CATEGORY, PAIR_KINDS, SPLITS, SUBSET, TARGETS

COLUMNS = {"seen": ("train", SUBSET, "dev", "test_seen", "reserve"), "unseen": ("test_unseen", "reserve")}
SHARE_COLUMNS = ("train", SUBSET, "dev", "test_seen", "test_unseen")
LABEL = {"train": "train", SUBSET: "subset human", "dev": "dev", "test_seen": "test_seen",
         "test_unseen": "test_unseen", "reserve": "reserve"}
COUNT_LABEL = {SUBSET: "train_human_subset"}
PAIR_LABEL = {"template_series": "Chuỗi template cùng nguồn (`template_series.csv`)",
              "dedup_review": "Cặp cần review (`dedup_review.csv`)",
              "dedup_blocked": "Liên kết bị chặn (`dedup_pairs.csv`, applied=False)"}


def num(n):
    """Vietnamese integer: thousands separated by '.'."""
    return f"{n:,}".replace(",", ".")


def dec(x, digits=2):
    """Vietnamese decimal: ',' as decimal mark."""
    return f"{x:.{digits}f}".replace(".", ",")


def _pct(part, whole):
    return f"{dec(100 * part / whole)}%" if whole else "–"


def cell_counts(manifest, reserve):
    """Counter[(source, category, column)] where column is a split, human_subset or reserve."""
    counts = Counter()
    for r in manifest:
        counts[(r["source"], r["category"], r["split"])] += 1
        if r["is_human_subset"]:
            counts[(r["source"], r["category"], SUBSET)] += 1
    for r in reserve:
        counts[(r["source"], r["category"], "reserve")] += 1
    return counts


def split_count_rows(sources, pool, manifest, reserve):
    counts = cell_counts(manifest, reserve)
    rows = []
    for source, row in sources.items():
        categories = sorted({r["category"] or NO_CATEGORY for r in pool if r["source"] == source})
        for category in categories:
            for column in COLUMNS[row["outlet_group"]]:
                rows.append({"source": source, "category": category, "split": COUNT_LABEL.get(column, column),
                             "records": counts[(source, category, column)]})
    return rows


def share_deviation(sources, pool, manifest, selectable):
    """{source: {column: max |share in column - share in selectable pool|}} over categories.

    Columns with no selected record (target 0) are left out."""
    counts = cell_counts(manifest, [])
    result = {}
    for source in sources:
        base = Counter(r["category"] for r in pool if r["source"] == source and r["record_id"] in selectable)
        total = sum(base.values())
        result[source] = {}
        for column in SHARE_COLUMNS:
            size = sum(counts[(source, c, column)] for c in base)
            if size and total:
                result[source][column] = max(abs(counts[(source, c, column)] / size - n / total)
                                             for c, n in base.items())
    return result


def quota_text(sources):
    """Quota sentence and human scope derived from sources.csv (sources with the same targets share a line)."""
    profiles = defaultdict(list)
    for source, row in sources.items():
        profiles[(row["outlet_group"], *(int(row[TARGETS[c]]) for c in (*SPLITS, SUBSET)))].append(source)
    parts = []
    for (outlet, train, dev, test_seen, test_unseen, subset), names in profiles.items():
        terms = [f"train {num(train)}" + (f" (subset human {num(subset)})" if subset else "")] if train else []
        terms += [f"{label} {num(n)}" for label, n in (("dev", dev), ("test_seen", test_seen),
                                                         ("test_unseen", test_unseen)) if n]
        who = f"mỗi nguồn {outlet} ({len(names)} nguồn: {', '.join(names)})"
        parts.append(f"{who}: {num(train + dev + test_seen + test_unseen)} = " + " + ".join(terms or ["0"]))
    scope = {c: sum(int(row[TARGETS[c]]) for row in sources.values()) for c in (SUBSET, "dev", "test_seen",
                                                                                 "test_unseen")}
    return ("Quota theo `configs/crawling/sources.csv`: " + "; ".join(parts) + f". Phạm vi human "
            f"{num(sum(scope.values()))} = subset {num(scope[SUBSET])} + dev {num(scope['dev'])} + test_seen "
            f"{num(scope['test_seen'])} + test_unseen {num(scope['test_unseen'])}.")


def bound_text(sources):
    """Rounding bound 1/quota (percentage points) per column, from the smallest non-zero target."""
    bounds = []
    for column in SHARE_COLUMNS:
        targets = [int(row[TARGETS[column]]) for row in sources.values() if int(row[TARGETS[column]])]
        if targets:
            bounds.append(f"{LABEL[column]} {dec(100 / min(targets), 3)}")
    return "; ".join(bounds)


def render(ctx):
    sources, pool, manifest, reserve = ctx["sources"], ctx["pool"], ctx["manifest"], ctx["reserve"]
    selectable = ctx["selectable"]
    counts = cell_counts(manifest, reserve)
    per_source = Counter()
    for (source, _, column), n in counts.items():
        per_source[(source, column)] += n
    seen = [s for s, row in sources.items() if row["outlet_group"] == "seen"]
    unseen = [s for s, row in sources.items() if row["outlet_group"] == "unseen"]
    null_rows = Counter(r["source"] for r in pool if r["category"] is None)
    held_rows = Counter(r["source"] for r in reserve if r["hold_reason"] == HOLD_UNSEEN)
    lines = ["# Báo cáo chia tập (split)", "",
             f"Sinh bởi `scripts/build_split.py`, seed **{ctx['seed']}** (`configs/crawling/run.json`). Không ghi thời "
             "điểm chạy để hai lần chạy cho output giống hệt từng byte. Số viết theo kiểu Việt Nam (nghìn `.`, thập "
             "phân `,`); các file CSV dùng số thô.", "",
             "Input (chỉ đọc):", ""]
    lines += [f"- `{name}`: SHA-256 `{digest}`" for name, digest in ctx["inputs"]]
    lines += ["", "## 1. Pool đủ điều kiện", "",
              "Đủ điều kiện = đại diện nhóm trùng (`is_dup_representative`) và không có `dedup_excluded_reason`. "
              "Mọi category khác null đều có mapping `approved` (script từ chối chạy nếu không). Không chọn được: bài "
              f"có category null (nằm trong reserve với stratum `{NO_CATEGORY}`, `hold_reason={HOLD_NO_CATEGORY}`) và "
              "bài seen có cặp gần trùng chưa review với bài unseen (`hold_reason="
              f"{HOLD_UNSEEN}`, mục 5); cả nhóm `split_group` chứa bài đó cũng không chọn được.", "",
              "| Nguồn | Nhóm | Bài hợp lệ | Đủ điều kiện | Có sapo | Category null | Giữ chờ review | Chọn được | "
              "Vào split | Reserve |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source, row in sources.items():
        mine = [r for r in pool if r["source"] == source]
        chosen = sum(per_source[(source, s)] for s in SPLITS)
        lines.append(f"| {source} | {row['outlet_group']} | {num(ctx['valid'][source])} | {num(len(mine))} | "
                     f"{num(sum(r['has_sapo'] for r in mine))} | {num(null_rows[source])} | {num(held_rows[source])} | "
                     f"{num(sum(r['record_id'] in selectable for r in mine))} | {num(chosen)} | "
                     f"{num(per_source[(source, 'reserve')])} |")
    lines += ["", f"Tổng: {num(sum(ctx['valid'].values()))} bài hợp lệ, {num(len(pool))} đủ điều kiện, "
              f"{num(len(selectable))} chọn được, {num(len(manifest))} vào split, {num(len(reserve))} reserve; "
              f"{num(sum(null_rows.values()))} bài category null "
              f"({', '.join(f'{s}: {num(n)}' for s, n in null_rows.items()) or 'không có'}); "
              f"{num(sum(held_rows.values()))} bài giữ chờ review dedup với bài unseen "
              f"({', '.join(f'{s}: {num(n)}' for s, n in held_rows.items()) or 'không có'}).", "",
              "## 2. Số bài theo nguồn × split", "",
              "| Nguồn | train | trong đó subset human | dev | test_seen | test_unseen | Tổng | Human (subset+dev+test) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    totals = Counter()
    keys = ("train", SUBSET, "dev", "test_seen", "test_unseen", "total", "human")
    for source in sources:
        values = {c: per_source[(source, c)] for c in ("train", SUBSET, "dev", "test_seen", "test_unseen")}
        values["total"] = values["train"] + values["dev"] + values["test_seen"] + values["test_unseen"]
        values["human"] = values["total"] - values["train"] + values[SUBSET]
        totals.update(values)
        lines.append(f"| {source} | " + " | ".join(num(values[c]) for c in keys) + " |")
    lines.append("| **Tổng** | " + " | ".join(f"**{num(totals[c])}**" for c in keys) + " |")
    lines += ["", quota_text(sources), "",
              "## 3. Cơ cấu category trong từng nguồn", "",
              "Quota mỗi split (và subset trong train, test_unseen) được chia theo tỷ lệ category của các bài chọn được "
              "của nguồn bằng phương pháp phần dư lớn nhất (hòa thì theo tên category). Không dùng nhãn nào. Độ lệch "
              "= max theo category của |tỷ lệ trong split − tỷ lệ trong pool chọn được|, đơn vị điểm phần trăm; giới "
              f"hạn do làm tròn là 1/quota ({bound_text(sources)}; tính với quota nhỏ nhất của mỗi cột).", "",
              "| Nguồn | train | subset human | dev | test_seen | test_unseen | Max |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source, deviation in ctx["deviation"].items():
        cells = [dec(100 * deviation[c]) if c in deviation else "–" for c in SHARE_COLUMNS]
        worst = f"**{dec(100 * max(deviation.values()))}**" if deviation else "–"
        lines.append(f"| {source} | " + " | ".join(cells) + f" | {worst} |")
    for group, names in (("seen", seen), ("unseen", unseen)):
        columns = COLUMNS[group]
        lines += ["", f"### Nguồn {group}: số bài theo category", "",
                  "| Nguồn | Category | Đủ điều kiện | Chọn được | Tỷ lệ chọn được | "
                  + " | ".join(LABEL[c] for c in columns) + " |",
                  "| --- | --- | ---: | ---: | ---: |" + " ---: |" * len(columns)]
        for source in names:
            mine = [r for r in pool if r["source"] == source]
            base = Counter(r["category"] or NO_CATEGORY for r in mine)
            chosen = Counter(r["category"] for r in mine if r["record_id"] in selectable)
            total = sum(chosen.values())
            for category in sorted(base):
                lines.append(f"| {source} | {category} | {num(base[category])} | {num(chosen[category])} | "
                             f"{_pct(chosen[category], total) if category != NO_CATEGORY else '–'} | "
                             + " | ".join(num(counts[(source, category, c)]) for c in columns) + " |")
    lines += ["", "## 4. Sapo theo split", "",
              "Dev bắt buộc có sapo (mục 3.1: so sánh title / title+sapo trên cùng bài); quota dev theo category "
              "được lấy chỉ từ bài có sapo. Train, subset và test không yêu cầu sapo.", "",
              "| Split | Bài | Có sapo | Tỷ lệ |", "| --- | ---: | ---: | ---: |"]
    has_sapo = {r["record_id"]: r["has_sapo"] for r in pool}
    groups = defaultdict(list)
    for r in manifest:
        groups[r["split"]].append(r["record_id"])
        if r["is_human_subset"]:
            groups[SUBSET].append(r["record_id"])
    groups["reserve"] = [r["record_id"] for r in reserve]
    for column in ("train", SUBSET, "dev", "test_seen", "test_unseen", "reserve"):
        ids = groups[column]
        with_sapo = sum(has_sapo[i] for i in ids)
        lines.append(f"| {LABEL[column]} | {num(len(ids))} | {num(with_sapo)} | {_pct(with_sapo, len(ids))} |")
    g = ctx["groups"]
    by_kind = g["pairs"]["by_kind"]
    lines += ["", "## 5. split_group (đơn vị lấy mẫu) và cặp gần trùng chưa gộp", "",
              "Bài cùng `dup_group` đã gộp thành một đại diện ở bước clean. Các cặp gần trùng mà bước clean liệt kê "
              "nhưng không gộp (chuỗi template cùng nguồn; cặp `dedup_review.csv` chưa có quyết định; liên kết bị "
              "chặn trong `dedup_pairs.csv`) được nối (union-find) thành một `split_group` nếu hai bài cùng phía "
              "seen/unseen, để hai tiêu đề gần giống hệt không bao giờ nằm ở hai split. Theo mặc định của nhóm, các "
              "bài này được giữ lại, không xóa; nối nhóm chỉ ràng buộc chúng vào cùng split. Cặp seen–unseen không "
              "thể chung split: nếu chưa có quyết định `separate` trong `configs/checking/dedup_decisions.csv`, bài "
              f"seen bị giữ trong reserve (`hold_reason={HOLD_UNSEEN}`) cho tới khi review dedup quyết định (`link` "
              "thì bước clean loại bài seen như `duplicate_of_unseen`). Quota tính theo bài; nhóm không vừa phần "
              "quota còn lại của mọi ô (nguồn, category) mà nó chạm tới thì bị bỏ qua, xét nhóm tiếp theo theo thứ "
              "tự seed.", "",
              "| Loại cặp | Dòng trong file | Cặp chưa gộp | Nối vào split_group | Seen–unseen: giữ bài seen | "
              "Seen–unseen: đã `separate` | Đã quyết định `separate` | Hai bài ở hai split |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in PAIR_KINDS:
        k = by_kind[kind]
        lines.append(f"| {PAIR_LABEL[kind]} | {num(k['listed'])} | {num(k['unmerged'])} | {num(k['linked'])} | "
                     f"{num(k['cross_outlet_held'])} | {num(k['cross_outlet_separate'])} | {num(k['separate'])} | "
                     f"{num(k['split_apart'])} |")
    lines += ["", "Một cặp nằm ở nhiều file được tính một lần, theo thứ tự template → review → bị chặn. Cột \"Hai bài ở "
              "hai split\" chỉ được khác 0 với cặp đã quyết định `separate` (kiểm tra ở mục 7). Dòng `dedup_review.csv` "
              "của liên kết đã áp dụng (`auto_link_check`, `exact_date_gap`) không phải cặp chưa gộp.", "",
              f"- {num(g['template_pairs'])} cặp template trong {num(g['template_series'])} chuỗi; {num(g['units'])} "
              f"split_group trong pool, trong đó **{num(g['multi'])} nhóm ≥2 bài** ({num(g['multi_records'])} bài).",
              f"- Kích thước nhóm nhiều bài: {', '.join(f'{k} bài: {num(v)}' for k, v in sorted(g['sizes'].items()))}.",
              f"- Nhóm nhiều bài trải nhiều category: {num(g['multi_category'])}; trải nhiều nguồn: "
              f"{num(g['multi_source'])}.",
              f"- `dup_group` seen bị giữ chờ review với bài unseen: {num(g['pairs']['held_dup_groups'])}.",
              "- Nơi nhận nhóm nhiều bài: " + ", ".join(f"{k}: {num(v[0])} nhóm/{num(v[1])} bài"
                                                        for k, v in sorted(g["placement"].items())) + "."]
    lines += ["", "## 6. Reserve", "",
              "`data/manifests/reserve_list.csv`: mọi bài đủ điều kiện không vào split, kèm `has_sapo` và "
              "`hold_reason`. `reserve_rank` tiếp nối thứ tự seed của từng (nguồn, category) (1 = bài thay thế đầu "
              "tiên). Dùng để thay bài bị loại khi gán nhãn (mục 2.4): lấy bài có `reserve_rank` nhỏ nhất cùng "
              "nguồn/category và `hold_reason` rỗng; **thay bài dev thì bài thay thế phải có `has_sapo=True`** (cả "
              "`split_group` của nó), bỏ qua bài không có sapo. Nếu bài thay thế thuộc `split_group` nhiều bài thì "
              "phải lấy cả nhóm hoặc bỏ qua nhóm đó. Không chọn theo nhãn LLM hay kết quả model. Số bài reserve theo "
              "nguồn:", "",
              "| Nguồn | Reserve | Có sapo | Có hold_reason | Dùng được | Category ít reserve dùng được nhất |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    usable = Counter((r["source"], r["category"]) for r in reserve if not r["hold_reason"])
    for source in sources:
        mine = [r for r in reserve if r["source"] == source]
        cells = {c: n for (s, c), n in usable.items() if s == source}
        cells.update({r["category"]: 0 for r in mine if (source, r["category"]) not in usable
                      and r["category"] != NO_CATEGORY})
        least = min(cells.items(), key=lambda kv: (kv[1], kv[0])) if cells else ("–", 0)
        lines.append(f"| {source} | {num(len(mine))} | {num(sum(r['has_sapo'] for r in mine))} | "
                     f"{num(sum(bool(r['hold_reason']) for r in mine))} | {num(sum(not r['hold_reason'] for r in mine))} "
                     f"| {least[0]}: {num(least[1])} |")
    lines += ["", "Số reserve theo từng category: cột reserve ở mục 3 và `reports/split_counts.csv`.", "",
              "## 7. Kiểm tra (đều đạt)", ""]
    lines += [f"- {text}" for text in ctx["checks"]]
    lines += ["", "## 8. Bảng blind và ánh xạ", "",
              f"- `data/annotations/blind/blind_dev.csv`: {num(ctx['blind_dev'])} dòng (dev; gán nhãn trước theo flow "
              "của nhóm).",
              f"- `data/annotations/blind/blind_train_subset_test.csv`: {num(ctx['blind_rest'])} dòng (subset train + "
              "test_seen + test_unseen).",
              "- Hai bảng chỉ có `annotation_id,title`; không có nguồn, URL, split, category hay nhãn LLM. Thứ tự dòng "
              "là một hoán vị seed trộn đều nguồn/category/split. Title giữ nguyên như clean; "
              f"{num(ctx['multiline_titles'])} title có xuống dòng bên trong (CSV đặt trong ngoặc kép, đọc bằng CSV "
              "reader).",
              "- `annotation_id` = `A` + 5 chữ số lấy ngẫu nhiên (seed) từ 10000–99999, không tuần tự theo nguồn, "
              "split hay `record_id`.",
              f"- `data/manifests/annotation_map.csv` (nội bộ, không gửi annotator): {num(ctx['mapped'])} dòng "
              "`annotation_id, record_id, split, source`.", "",
              "## 9. Việc còn mở", "",
              "- `annotation_assignment.csv` (mục 2.2: 9 chunk C1..C9, người P1..P9, lô ~100 mẫu) do P3 (lead human "
              "annotation) tạo từ `annotation_map.csv` và hai bảng blind; bước split không tạo file này.",
              f"- Review dedup: {num(by_kind['dedup_review']['unmerged'] - by_kind['dedup_review']['separate'])} cặp "
              "`dedup_review.csv` chưa có quyết định "
              "(đã được giữ cùng split hoặc giữ bài seen trong reserve, nên không rò rỉ giữa split). Nên review "
              "(ghi `link`/`separate` vào `configs/checking/dedup_decisions.csv`) **trước khi bắt đầu gán nhãn**: "
              "chạy lại `build_clean.py` + `build_split.py` sau khi đổi quyết định sẽ đổi split.",
              "- Quy trình thay bài từ reserve khi gán nhãn (cấp `annotation_id` mới cho bài thay thế, cập nhật "
              "manifest) chưa tự động hóa.",
              f"- Bài category null ({num(sum(null_rows.values()))}) chỉ nằm trong reserve; cần map category nếu muốn "
              "dùng.",
              ""]
    return "\n".join(lines)
