"""Render reports/data_report.md: raw crawl (from crawl_stats.csv) and clean sections."""

from __future__ import annotations

import json
from collections import Counter

POLICIES = [
    "Raw quota: 3,500 articles per seen outlet, 500 per unseen outlet; 01/01/2026–21/10/2026 first, falling back "
    "to 01/01/2024 only on shortfall.",
    "Inventory traverses every sitemap/feed, then shuffles with seed 2026; no per-category quota.",
    "Only the plan's 9 fields are stored; no body/HTML/media. Rows missing sapo/category are kept (partial); "
    "SEO description is never used as sapo.",
    "CafeF: timestamps without an offset are treated as UTC+07:00 (team-approved).",
    "VietNamNet: 13 sitemap records (11 URLs from 2022) with control characters in the URL are quarantined; "
    "no URL is rewritten.",
    "Tag sitemaps (`sitemap-tags-NN.xml`) and tag-page URLs `-tagNNN.html` are skipped (2sao, VietNamNet).",
    "Dân Trí: the 'Dân trí 360' format (JS-rendered short video, no title/date in HTML) and 5 emagazine articles "
    "with image titles are systematically excluded (team decision); they show up as missing_title.",
    "Dân Trí: emagazine sapo is re-fetched separately with a new selector and accepted only when the re-fetched "
    "title matches the stored title. `scripts/export_raw.py` fills it into `articles_raw.jsonl` (rows without a "
    "sapo only) and recomputes `extraction_status` with the crawler's rule (`success` when title, sapo, category "
    "and published_at are all present, otherwise `partial`). Crawl files in `data/crawl_work/` are not modified; "
    "filled record_ids are listed in `sapo_refetch/<refetch_id>/sapo_supplement.jsonl` (status `sapo_recovered`) "
    "in the run directory.",
    "Dân Trí rate-limits at ~100 requests/3 minutes (HTTP 428): 2.5 s delay and a ≥120 s wait on 428.",
    "YAN has no separate sapo block; only a fully bold first paragraph is taken as sapo.",
]
CRAWL_LIMITATIONS = [
    "The raw quota is not a clean quota: see the Clean section (checking, dedup, funnel).",
    "A complete inventory only means every discovered endpoint was traversed; it does not prove coverage of "
    "every published article.",
]
LIMITATIONS = [
    "Downloaded = distinct `record_id`s in the crawl run's `articles.jsonl` (all attempts, taken from "
    "`reports/crawl_stats.csv`) with `extraction_status` success/partial and a non-empty title; includes articles "
    "outside the date range or not selected into the raw quota. success/partial rows without a title "
    "(listing/tag pages, e.g. 3,739 2sao 'Trang chủ' not_article_page rows, and missing_title rows) are reported "
    "separately as `downloaded_without_title`.",
    "Valid = selected articles that pass record-level checking; after_dedup = non-excluded duplicate-group "
    "representatives; with_sapo is counted within after_dedup.",
    "`<unmapped>` in the downloaded column of `funnel_by_source_category.csv` is mostly downloaded but unselected "
    "articles whose raw category is not in the mapping (the mapping is proposed from selected articles).",
    "Dedup compares titles only (char-3gram Jaccard); sapo similarity only supports review and never links.",
    "Auto-link precision has not been measured. Near-duplicate links require |published_at gap| <= "
    "near_link_max_days; applied near-duplicate links with JC3 < review_auto_link_below and exact-title links "
    "outside the date window are listed in `dedup_review.csv` (auto_linked=True) for spot checks.",
    "A link is refused when it would put a number-conflict pair or a manual `separate` pair in the same group "
    "(transitive_conflict); the result depends on link order (exact matches first, then descending JC3), which "
    "is fixed.",
    "Same-source template series are only reported, not merged into dup_group (pending team decision).",
    "Category mapping rows stay proposed/needs_review until team review; unmapped keys stay null and "
    "`category_mapping_status` is null when category is null.",
    "Mojibake detection is pattern-based; only titles with mojibake confirmed by round-trip are rejected.",
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
    lines = ["## 1. Raw crawl", "",
             "Source: `reports/crawl_stats.csv` (written by `scripts/export_raw.py` with `data/raw/articles_raw.jsonl`).", "",
             f"- Selected into the raw quota: **{total['selected']:,}/{total['target']:,}**; shortfall **{total['shortfall']:,}**.",
             f"- Publication year: {', '.join(f'{y}: {n:,}' for y, n in sorted(years.items()))}.",
             f"- Article URL attempts {total['attempted_urls']:,}; downloaded (with title) {total['downloaded']:,}; "
             f"success/partial without title {total['downloaded_without_title']:,}.",
             f"- `extraction_status` in raw: `success` {total['success']:,}, `partial` {total['partial']:,}.",
             f"- Missing sapo after crawl {total['missing_sapo_before_refetch']:,}; recovered by refetch "
             f"{total['sapo_refetched']:,}; still missing in raw **{total['missing_sapo']:,}** "
             f"({100 * total['missing_sapo'] / max(total['selected'], 1):.2f}%). Missing category {total['missing_category']:,}.",
             "", "| Source | Group | Run | Inventory URLs | Attempts | Downloaded | No title | Selected | Target | Shortfall | "
             "Missing sapo (crawl) | Refetched | Missing sapo (raw) | Missing category | Raw categories | Published |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in crawl_rows:
        lines.append(f"| {r['source']} | {r['outlet_group']} | `{r['run_id']}` | {_n(r['inventory_urls'] or 0)} | "
                     f"{_n(r['attempted_urls'])} | {_n(r['downloaded'])} | {_n(r['downloaded_without_title'])} | "
                     f"{_n(r['selected'])} | {_n(r['target'])} | {_n(r['shortfall'])} | "
                     f"{_n(r['missing_sapo_before_refetch'])} | {_n(r['sapo_refetched'])} | {_n(r['missing_sapo'])} | "
                     f"{_n(r['missing_category'])} | {r['raw_categories']} | {r['date_min'][:10]} → {r['date_max'][:10]} |")
    lines += ["", "Selection outcome over all attempts (crawler `selection_status`):", "",
              "| Source | " + " | ".join(s.removeprefix("selection_") for s in statuses) + " |",
              "| --- |" + " ---: |" * len(statuses)]
    lines += [f"| {r['source']} | " + " | ".join(_n(r[s]) for s in statuses) + " |" for r in crawl_rows]
    lines += ["", "### Policies and decisions", ""] + [f"- {p}" for p in POLICIES]
    lines += ["", "### Limitations", ""] + [f"- {item}" for item in CRAWL_LIMITATIONS]
    return lines


def clean_section(summary, by_source, rejection_counts):
    lines = ["## 2. Clean", "",
             "Input: `data/raw/articles_raw.jsonl`; the parquet `source_run_id` column comes from `run_id` in "
             "`reports/crawl_stats.csv`.", "",
             "### Funnel: downloaded → valid → after dedup → with sapo", "",
             "| Source | Group | Downloaded | No title* | Selected | Valid | Rejected | Dedup excluded | After dedup | With sapo | Clean target | Shortfall |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in by_source:
        lines.append(f"| {r['source']} | {r['outlet_group']} | {r['downloaded']:,} | {r['downloaded_without_title']:,} | "
                     f"{r['selected']:,} | {r['valid']:,} | "
                     f"{r['rejected']:,} | {r['dedup_excluded']:,} | {r['after_dedup']:,} | {r['with_sapo']:,} | "
                     f"{r['min_clean_articles']:,} | {r['shortfall']:,} |")
    t = summary["totals"]
    lines += ["", f"Total: downloaded **{t['downloaded']:,}**, selected {t['selected']:,}, valid **{t['valid']:,}**, "
              f"after dedup **{t['after_dedup']:,}**, with sapo **{t['with_sapo']:,}**.", "",
              "Downloaded = distinct `record_id`s with `extraction_status` success/partial **and a title** over all "
              "attempts of the crawl run (including articles outside the date range or not selected). *No title = "
              "success/partial rows without a title (tag/listing pages, e.g. 2sao `Trang chủ` not_article_page; "
              "missing_title); not counted as downloaded.", "", "### Rejected articles", ""]
    if rejection_counts:
        lines += ["| Source | Issue | Articles |", "| --- | --- | ---: |"]
        lines += [f"| {r['source']} | {r['issue']} | {r['records']} |" for r in rejection_counts]
    else:
        lines.append("No rejected articles.")
    lines += ["", "Issues by type and decision (`data/interim/issues.csv`): "
              + ", ".join(f"`{k}` {v:,}" for k, v in summary["issue_counts"].items()) + ".",
              "Technical fixes by field (`data/interim/fixes.csv`): "
              + (", ".join(f"`{k}` {v:,}" for k, v in summary["fix_counts"].items()) or "none") + "."]
    d = summary["dedup"]
    lines += ["", "### Dedup", "",
              f"- {d['groups']:,} dup_groups; {d['multi_member_groups']:,} groups with ≥2 articles "
              f"({d['cross_source_groups']:,} cross-source), largest {d['largest_group']} articles.",
              f"- Auto links: {d['auto_links']:,} pairs ({', '.join(f'{k}: {v}' for k, v in d['auto_link_rules'].items())}).",
              f"- Excluded after dedup: {d['excluded']:,} articles; `duplicate_of_unseen` "
              f"{d['excluded_duplicate_of_unseen']:,}, `duplicate_in_group` {d['excluded_duplicate_in_group']:,}.",
              f"- Blocked: {d['blocked_links']:,} links ({', '.join(f'{k}: {v}' for k, v in d['blocked_reasons'].items()) or 'none'}); "
              "`transitive_conflict` = a link that would merge two number-conflicting articles (or a `separate` pair) "
              "through a third.",
              f"- Needs review (`dedup_review.csv`): {d['review_pairs']:,} pairs "
              f"({', '.join(f'{k}: {v}' for k, v in d['review_rules'].items())}). `auto_link_check`/`exact_date_gap` "
              "are applied links to verify; auto-link precision has not been measured.",
              f"- Same-source template series: {d['template_pairs']:,} pairs in {d['template_series']:,} series "
              "(`template_series.csv`, not merged)."]
    lines += [f"- Note: {note}" for note in d["notes"]]
    c = summary["category"]
    lines += ["", "### Category", "",
              f"- Valid articles unmapped: {c['valid_unmapped']:,}; unapproved mapping (proposed/needs_review): "
              f"{c['valid_unapproved_mapping']:,} ({', '.join(f'{k}: {v:,}' for k, v in c['mapping_status_counts'].items())}).",
              f"- Mapping `{summary['configs']['mapping']}`, {summary['configs']['dedup_decisions']} manual dedup decisions.",
              "- Shared categories are not yet team-approved; the split step must refuse mappings that are not `approved`.",
              "", "### Limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    return lines


def render(created_at, crawl_rows, summary, by_source, rejection_counts):
    lines = ["# Data report", "", f"Generated at {created_at} by `scripts/build_clean.py`.", ""]
    lines += crawl_section(crawl_rows) + [""] + clean_section(summary, by_source, rejection_counts)
    lines += ["", "## Files", "",
              f"- `data/raw/articles_raw.jsonl`: {sum(int(r['selected']) for r in crawl_rows):,} selected raw articles, "
              "exactly the plan's 9 fields.",
              "- `reports/crawl_stats.csv`: per-source crawl stats (`downloaded_by_category` is JSON "
              "`[[raw category, articles], ...]` used for the per-category funnel).",
              "- `reports/funnel_by_source.csv`, `reports/funnel_by_source_category.csv` (`<unmapped>` category shown separately).",
              "- `data/processed/articles_clean.parquet`: valid articles with `dup_group`.",
              "- `data/interim/`: `issues.csv`, `rejected.csv`, `fixes.csv`, `dedup_pairs.csv`, `dedup_review.csv`, "
              "`template_series.csv`.", ""]
    return "\n".join(lines)
