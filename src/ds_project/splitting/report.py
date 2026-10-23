"""reports/split_report.md and reports/split_counts.csv (no timestamp, so reruns are byte-identical)."""

from __future__ import annotations

from collections import Counter, defaultdict

from .split import HOLD_NO_CATEGORY, HOLD_UNSEEN, NO_CATEGORY, PAIR_KINDS, SPLITS, SUBSET, TARGETS

COLUMNS = {"seen": ("train", SUBSET, "dev", "test_seen", "reserve"), "unseen": ("test_unseen", "reserve")}
SHARE_COLUMNS = ("train", SUBSET, "dev", "test_seen", "test_unseen")
LABEL = {"train": "train", SUBSET: "human subset", "dev": "dev", "test_seen": "test_seen",
         "test_unseen": "test_unseen", "reserve": "reserve"}
COUNT_LABEL = {SUBSET: "train_human_subset"}
PAIR_LABEL = {"template_series": "Same-source template series (`template_series.csv`)",
              "dedup_review": "Pairs to review (`dedup_review.csv`)",
              "dedup_blocked": "Blocked links (`dedup_pairs.csv`, applied=False)"}


def num(n):
    """Integer with ',' as thousands separator."""
    return f"{n:,}"


def dec(x, digits=2):
    return f"{x:.{digits}f}"


def _pct(part, whole):
    return f"{dec(100 * part / whole)}%" if whole else "–"


def cell_counts(manifest, reserve):
    """Counter[(source, category, column)]; column is a split, human_subset or reserve."""
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
    """{source: {column: max over categories of |share - selectable-pool share|}}; empty columns omitted."""
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
    """Quota sentence from sources.csv; sources with the same targets are listed together."""
    profiles = defaultdict(list)
    for source, row in sources.items():
        profiles[(row["outlet_group"], *(int(row[TARGETS[c]]) for c in (*SPLITS, SUBSET)))].append(source)
    parts = []
    for (outlet, train, dev, test_seen, test_unseen, subset), names in profiles.items():
        terms = [f"train {num(train)}" + (f" (human subset {num(subset)})" if subset else "")] if train else []
        terms += [f"{label} {num(n)}" for label, n in (("dev", dev), ("test_seen", test_seen),
                                                         ("test_unseen", test_unseen)) if n]
        who = f"each {outlet} source ({len(names)} sources: {', '.join(names)})"
        parts.append(f"{who}: {num(train + dev + test_seen + test_unseen)} = " + " + ".join(terms or ["0"]))
    scope = {c: sum(int(row[TARGETS[c]]) for row in sources.values()) for c in (SUBSET, "dev", "test_seen",
                                                                                 "test_unseen")}
    return ("Quotas from `configs/crawling/sources.csv`: " + "; ".join(parts) + f". Human scope "
            f"{num(sum(scope.values()))} = subset {num(scope[SUBSET])} + dev {num(scope['dev'])} + test_seen "
            f"{num(scope['test_seen'])} + test_unseen {num(scope['test_unseen'])}.")


def bound_text(sources):
    """Rounding bound 1/quota (percentage points) per column, from its smallest non-zero target."""
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
    lines = ["# Split report", "",
             f"Generated by `scripts/build_split.py`, seed **{ctx['seed']}** (`configs/crawling/run.json`). No run "
             "timestamp, so two runs give byte-identical output. The CSV files use raw numbers.", "",
             "Inputs (read only):", ""]
    lines += [f"- `{name}`: SHA-256 `{digest}`" for name, digest in ctx["inputs"]]
    lines += ["", "## 1. Eligible pool", "",
              "Eligible = duplicate-group representative (`is_dup_representative`) with no `dedup_excluded_reason`. "
              "Every non-null category has an `approved` mapping (the script refuses to run otherwise). Not "
              f"selectable: null-category records (kept in reserve with stratum `{NO_CATEGORY}`, "
              f"`hold_reason={HOLD_NO_CATEGORY}`) and seen records with an unreviewed near-duplicate pair with an "
              f"unseen record (`hold_reason={HOLD_UNSEEN}`, section 5); their whole `split_group` is not selectable "
              "either.", "",
              "| Source | Group | Valid | Eligible | With sapo | Null category | Held for review | Selectable | "
              "In splits | Reserve |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source, row in sources.items():
        mine = [r for r in pool if r["source"] == source]
        chosen = sum(per_source[(source, s)] for s in SPLITS)
        lines.append(f"| {source} | {row['outlet_group']} | {num(ctx['valid'][source])} | {num(len(mine))} | "
                     f"{num(sum(r['has_sapo'] for r in mine))} | {num(null_rows[source])} | {num(held_rows[source])} | "
                     f"{num(sum(r['record_id'] in selectable for r in mine))} | {num(chosen)} | "
                     f"{num(per_source[(source, 'reserve')])} |")
    lines += ["", f"Total: {num(sum(ctx['valid'].values()))} valid, {num(len(pool))} eligible, "
              f"{num(len(selectable))} selectable, {num(len(manifest))} in splits, {num(len(reserve))} reserve; "
              f"{num(sum(null_rows.values()))} null-category records "
              f"({', '.join(f'{s}: {num(n)}' for s, n in null_rows.items()) or 'none'}); "
              f"{num(sum(held_rows.values()))} records held for dedup review with an unseen record "
              f"({', '.join(f'{s}: {num(n)}' for s, n in held_rows.items()) or 'none'}).", "",
              "## 2. Records per source × split", "",
              "| Source | train | of which human subset | dev | test_seen | test_unseen | Total | "
              "Human (subset+dev+test) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    totals = Counter()
    keys = ("train", SUBSET, "dev", "test_seen", "test_unseen", "total", "human")
    for source in sources:
        values = {c: per_source[(source, c)] for c in ("train", SUBSET, "dev", "test_seen", "test_unseen")}
        values["total"] = values["train"] + values["dev"] + values["test_seen"] + values["test_unseen"]
        values["human"] = values["total"] - values["train"] + values[SUBSET]
        totals.update(values)
        lines.append(f"| {source} | " + " | ".join(num(values[c]) for c in keys) + " |")
    lines.append("| **Total** | " + " | ".join(f"**{num(totals[c])}**" for c in keys) + " |")
    lines += ["", quota_text(sources), "",
              "## 3. Category mix per source", "",
              "Each split quota (and the subset inside train, and test_unseen) is divided over the source's "
              "selectable category shares by largest remainder (ties by category name). No labels are used. "
              "Deviation = max over categories of |share in split − share in selectable pool|, in percentage points; "
              f"the rounding bound is 1/quota ({bound_text(sources)}; using the smallest quota in each column).", "",
              "| Source | train | human subset | dev | test_seen | test_unseen | Max |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source, deviation in ctx["deviation"].items():
        cells = [dec(100 * deviation[c]) if c in deviation else "–" for c in SHARE_COLUMNS]
        worst = f"**{dec(100 * max(deviation.values()))}**" if deviation else "–"
        lines.append(f"| {source} | " + " | ".join(cells) + f" | {worst} |")
    pooled = Counter()
    for (_, category, column), n in counts.items():
        pooled[(category, column)] += n
    sizes = {c: sum(n for (_, column), n in pooled.items() if column == c) for c in SHARE_COLUMNS}
    categories = sorted({category for category, column in pooled if column in SHARE_COLUMNS},
                        key=lambda c: (-pooled[(c, "train")], c))
    lines += ["", "### All sources: category share per split", "",
              "Pooled over sources. Seen splits share one mix because each seen source is stratified alone; "
              "test_unseen follows the unseen outlets' own mix.", "",
              "| Category | " + " | ".join(f"{LABEL[c]} ({num(sizes[c])})" for c in SHARE_COLUMNS) + " |",
              "| --- |" + " ---: |" * len(SHARE_COLUMNS)]
    for category in categories:
        lines.append(f"| {category} | " + " | ".join(_pct(pooled[(category, c)], sizes[c]) for c in SHARE_COLUMNS)
                     + " |")
    for group, names in (("seen", seen), ("unseen", unseen)):
        columns = COLUMNS[group]
        lines += ["", f"### {group.capitalize()} sources: records per category", "",
                  "| Source | Category | Eligible | Selectable | Selectable share | "
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
    lines += ["", "## 4. Sapo per split", "",
              "Dev requires a sapo (plan 3.1: title vs title+sapo on the same record); dev category quotas are "
              "filled only from records with a sapo. Train, subset and test do not require one.", "",
              "| Split | Records | With sapo | Share |", "| --- | ---: | ---: | ---: |"]
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
    lines += ["", "## 5. split_group (sampling unit) and unmerged near-duplicate pairs", "",
              "Records sharing a `dup_group` were merged into one representative by the clean step. Near-duplicate "
              "pairs the clean step listed but did not merge (same-source template series; undecided "
              "`dedup_review.csv` pairs; blocked links in `dedup_pairs.csv`) are joined (union-find) into one "
              "`split_group` when both records are on the same seen/unseen side, so two near-identical titles never "
              "land in two splits. By team default these records are kept, not deleted; joining only binds them to "
              "the same split. A seen–unseen pair cannot share a split: without a `separate` decision in "
              "`configs/checking/dedup_decisions.csv`, the seen record is held in reserve "
              f"(`hold_reason={HOLD_UNSEEN}`) until the dedup review decides (`link` makes the clean step exclude "
              "the seen record as `duplicate_of_unseen`). Quotas count records; a group that does not fit the "
              "remaining quota of every (source, category) cell it touches is skipped and the next group in seed "
              "order is tried.", "",
              "| Pair kind | Rows in file | Unmerged pairs | Joined into split_group | Seen–unseen: seen held | "
              "Seen–unseen: `separate` | Decided `separate` | Records in two splits |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in PAIR_KINDS:
        k = by_kind[kind]
        lines.append(f"| {PAIR_LABEL[kind]} | {num(k['listed'])} | {num(k['unmerged'])} | {num(k['linked'])} | "
                     f"{num(k['cross_outlet_held'])} | {num(k['cross_outlet_separate'])} | {num(k['separate'])} | "
                     f"{num(k['split_apart'])} |")
    lines += ["", "A pair listed in several files is counted once, in the order template → review → blocked. The "
              "\"Records in two splits\" column may be non-zero only for pairs decided `separate` (checked in "
              "section 7). `dedup_review.csv` rows of applied links (`auto_link_check`, `exact_date_gap`) are not "
              "unmerged pairs.", "",
              f"- {num(g['template_pairs'])} template pairs in {num(g['template_series'])} series; {num(g['units'])} "
              f"split_groups in the pool, of which **{num(g['multi'])} groups with ≥2 records** "
              f"({num(g['multi_records'])} records).",
              "- Multi-record group sizes: "
              f"{', '.join(f'{k} records: {num(v)}' for k, v in sorted(g['sizes'].items()))}.",
              f"- Multi-record groups spanning several categories: {num(g['multi_category'])}; several sources: "
              f"{num(g['multi_source'])}.",
              f"- Seen `dup_group`s held for review with an unseen record: {num(g['pairs']['held_dup_groups'])}.",
              "- Placement of multi-record groups: " + ", ".join(f"{k}: {num(v[0])} groups/{num(v[1])} records"
                                                                 for k, v in sorted(g["placement"].items())) + "."]
    lines += ["", "## 6. Reserve", "",
              "`data/manifests/reserve_list.csv`: every eligible record not in a split, with `has_sapo` and "
              "`hold_reason`. `reserve_rank` continues the seed order of each (source, category) (1 = first "
              "replacement). Used to replace records dropped during annotation (plan 2.4): take the lowest "
              "`reserve_rank` with the same source/category and an empty `hold_reason`; **a dev replacement must "
              "have `has_sapo=True`** (its whole `split_group`), skip records without sapo. If the replacement "
              "belongs to a multi-record `split_group`, take the whole group or skip it. Never choose by LLM labels "
              "or model output. Reserve records per source:", "",
              "| Source | Reserve | With sapo | With hold_reason | Usable | Category with fewest usable reserve |",
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
    lines += ["", "Reserve per category: the reserve column in section 3 and `reports/split_counts.csv`.", "",
              "## 7. Checks (all passed)", ""]
    lines += [f"- {text}" for text in ctx["checks"]]
    lines += ["", "## 8. Blind tables and mapping", "",
              f"- `data/annotations/blind/blind_dev.csv`: {num(ctx['blind_dev'])} rows (dev; annotated first, per "
              "the team flow).",
              f"- `data/annotations/blind/blind_train_subset_test.csv`: {num(ctx['blind_rest'])} rows (train subset + "
              "test_seen + test_unseen).",
              "- Both tables hold only `annotation_id,title,category` (shared approved category); no source, URL, "
              "split or LLM label. Row order is a seeded permutation mixing source/category/split. Titles are kept "
              f"as in clean; titles with internal line breaks: {num(ctx['multiline_titles'])} (quoted in the CSV, "
              "read with a CSV reader).",
              "- `annotation_id` = `A` + 5 digits sampled (seeded) from 10000–99999, not sequential by source, "
              "split or `record_id`.",
              f"- `data/manifests/annotation_map.csv` (internal, not sent to annotators): {num(ctx['mapped'])} rows "
              "`annotation_id, record_id, split, source`.", "",
              "## 9. Open items", "",
              "- `annotation_assignment.csv` (plan 2.2: 9 chunks C1..C9, people P1..P9, batches of ~100) is built "
              "by P3 (human annotation lead) from `annotation_map.csv` and the two blind tables; the split step "
              "does not create it.",
              f"- Dedup review: {num(by_kind['dedup_review']['unmerged'] - by_kind['dedup_review']['separate'])} "
              "undecided `dedup_review.csv` pairs (kept in the same split or with the seen record held in reserve, "
              "so nothing leaks across splits). Review them (write `link`/`separate` to "
              "`configs/checking/dedup_decisions.csv`) **before annotation starts**: re-running `build_clean.py` + "
              "`build_split.py` after changing decisions changes the split.",
              "- Replacing records from reserve during annotation (new `annotation_id` for the replacement, "
              "manifest update) is not automated yet.",
              f"- Null-category records ({num(sum(null_rows.values()))}) stay in reserve only; map their category "
              "to use them.",
              ""]
    return "\n".join(lines)
