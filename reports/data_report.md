# Data report

Generated at 2026-10-23T08:25:06+07:00 by `scripts/build_clean.py`.

## 1. Raw crawl

Source: `reports/crawl_stats.csv` (written by `scripts/export_raw.py` with `data/raw/articles_raw.jsonl`).

- Selected into the raw quota: **30,000/30,000**; shortfall **0**.
- Publication year: 2026: 30,000.
- Article URL attempts 39,798; downloaded (with title) 33,562; success/partial without title 3,891.
- `extraction_status` in raw: `success` 29,467, `partial` 533.
- Missing sapo after crawl 839; recovered by refetch 310; still missing in raw **529** (1.76%). Missing category 4.

| Source | Group | Run | Inventory URLs | Attempts | Downloaded | No title | Selected | Target | Shortfall | Missing sapo (crawl) | Refetched | Missing sapo (raw) | Missing category | Raw categories | Published |
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

Selection outcome over all attempts (crawler `selection_status`):

| Source | primary | fallback | not_article_page | missing_title | extraction_error | publication_date_unknown | before_fallback_date |
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

### Policies and decisions

- Raw quota: 3,500 articles per seen outlet, 500 per unseen outlet; 01/01/2026–21/10/2026 first, falling back to 01/01/2024 only on shortfall.
- Inventory traverses every sitemap/feed, then shuffles with seed 2026; no per-category quota.
- Only the plan's 9 fields are stored; no body/HTML/media. Rows missing sapo/category are kept (partial); SEO description is never used as sapo.
- CafeF: timestamps without an offset are treated as UTC+07:00 (team-approved).
- VietNamNet: 13 sitemap records (11 URLs from 2022) with control characters in the URL are quarantined; no URL is rewritten.
- Tag sitemaps (`sitemap-tags-NN.xml`) and tag-page URLs `-tagNNN.html` are skipped (2sao, VietNamNet).
- Dân Trí: the 'Dân trí 360' format (JS-rendered short video, no title/date in HTML) and 5 emagazine articles with image titles are systematically excluded (team decision); they show up as missing_title.
- Dân Trí: emagazine sapo is re-fetched separately with a new selector and accepted only when the re-fetched title matches the stored title. `scripts/export_raw.py` fills it into `articles_raw.jsonl` (rows without a sapo only) and recomputes `extraction_status` with the crawler's rule (`success` when title, sapo, category and published_at are all present, otherwise `partial`). Crawl files in `data/crawl_work/` are not modified; filled record_ids are listed in `sapo_refetch/<refetch_id>/sapo_supplement.jsonl` (status `sapo_recovered`) in the run directory.
- Dân Trí rate-limits at ~100 requests/3 minutes (HTTP 428): 2.5 s delay and a ≥120 s wait on 428.
- YAN has no separate sapo block; only a fully bold first paragraph is taken as sapo.

### Limitations

- The raw quota is not a clean quota: see the Clean section (checking, dedup, funnel).
- A complete inventory only means every discovered endpoint was traversed; it does not prove coverage of every published article.

## 2. Clean

Input: `data/raw/articles_raw.jsonl`; the parquet `source_run_id` column comes from `run_id` in `reports/crawl_stats.csv`.

### Funnel: downloaded → valid → after dedup → with sapo

| Source | Group | Downloaded | No title* | Selected | Valid | Rejected | Dedup excluded | After dedup | With sapo | Clean target | Shortfall |
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

Total: downloaded **33,562**, selected 30,000, valid **29,777**, after dedup **29,434**, with sapo **29,068**.

Downloaded = distinct `record_id`s with `extraction_status` success/partial **and a title** over all attempts of the crawl run (including articles outside the date range or not selected). *No title = success/partial rows without a title (tag/listing pages, e.g. 2sao `Trang chủ` not_article_page; missing_title); not counted as downloaded.

### Rejected articles

| Source | Issue | Articles |
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

Issues by type and decision (`data/interim/issues.csv`): `category_unmapped:report` 4, `missing_sapo:report` 370, `missing_sapo_policy:reject` 159, `mojibake_suspected:flag` 2, `non_headline_title:reject` 64.
Technical fixes by field (`data/interim/fixes.csv`): `sapo` 51, `category_raw` 460, `title` 30.

### Dedup

- 29,434 dup_groups; 341 groups with ≥2 articles (335 cross-source), largest 3 articles.
- Auto links: 344 pairs (exact_norm_title: 294, cross_source_jc3: 50).
- Excluded after dedup: 343 articles; `duplicate_of_unseen` 23, `duplicate_in_group` 320.
- Blocked: 2 links (separate_decision: 2); `transitive_conflict` = a link that would merge two number-conflicting articles (or a `separate` pair) through a third.
- Needs review (`dedup_review.csv`): 209 pairs (auto_link_check: 29, date_gap: 31, exact_date_gap: 5, number_conflict: 30, review_band: 114). `auto_link_check`/`exact_date_gap` are applied links to verify; auto-link precision has not been measured.
- Same-source template series: 521 pairs in 91 series (`template_series.csv`, not merged).

### Category

- Valid articles unmapped: 4; unapproved mapping (proposed/needs_review): 0 (approved: 29,773, unmapped: 4).
- Mapping `configs/crawling/category_mapping.csv`, 2 manual dedup decisions.
- Shared categories are not yet team-approved; the split step must refuse mappings that are not `approved`.

### Limitations

- Downloaded = distinct `record_id`s in the crawl run's `articles.jsonl` (all attempts, taken from `reports/crawl_stats.csv`) with `extraction_status` success/partial and a non-empty title; includes articles outside the date range or not selected into the raw quota. success/partial rows without a title (listing/tag pages, e.g. 3,739 2sao 'Trang chủ' not_article_page rows, and missing_title rows) are reported separately as `downloaded_without_title`.
- Valid = selected articles that pass record-level checking; after_dedup = non-excluded duplicate-group representatives; with_sapo is counted within after_dedup.
- `<unmapped>` in the downloaded column of `funnel_by_source_category.csv` is mostly downloaded but unselected articles whose raw category is not in the mapping (the mapping is proposed from selected articles).
- Dedup compares titles only (char-3gram Jaccard); sapo similarity only supports review and never links.
- Auto-link precision has not been measured. Near-duplicate links require |published_at gap| <= near_link_max_days; applied near-duplicate links with JC3 < review_auto_link_below and exact-title links outside the date window are listed in `dedup_review.csv` (auto_linked=True) for spot checks.
- A link is refused when it would put a number-conflict pair or a manual `separate` pair in the same group (transitive_conflict); the result depends on link order (exact matches first, then descending JC3), which is fixed.
- Same-source template series are only reported, not merged into dup_group (pending team decision).
- Category mapping rows stay proposed/needs_review until team review; unmapped keys stay null and `category_mapping_status` is null when category is null.
- Mojibake detection is pattern-based; only titles with mojibake confirmed by round-trip are rejected.

## Files

- `data/raw/articles_raw.jsonl`: 30,000 selected raw articles, exactly the plan's 9 fields.
- `reports/crawl_stats.csv`: per-source crawl stats (`downloaded_by_category` is JSON `[[raw category, articles], ...]` used for the per-category funnel).
- `reports/funnel_by_source.csv`, `reports/funnel_by_source_category.csv` (`<unmapped>` category shown separately).
- `data/processed/articles_clean.parquet`: valid articles with `dup_group`.
- `data/interim/`: `issues.csv`, `rejected.csv`, `fixes.csv`, `dedup_pairs.csv`, `dedup_review.csv`, `template_series.csv`.
