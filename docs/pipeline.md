# Pipeline: crawl → export raw → clean → split

How to run each step and the rules each step applies. File schemas are in
[data_contract.md](data_contract.md).

## Data layout

| Location | Contents |
| --- | --- |
| `data/crawl_work/<source>/<run_id>/` | Crawler working directory (temporary, not committed): every attempt, inventory, cache, and the resume journal. **Currently compressed** into the archive outside the repo (see below). |
| `data/crawl_work/_batches/<run_id>/` | Batch `status.json` and `<source>.console.log`. |
| `../ds-project-archive/` (outside the repo, only on the crawling machine) | `crawl_work_final_runs_20261023.zip` (the 12 final runs plus the sapo refetch; needed to rerun `export_raw.py`) and `crawl_archive_20261023.zip` (pilot, older runs, old logs and reports; for traceability only). |
| `data/raw/articles_raw.jsonl` | **The only raw file**: 30,000 selected articles, exactly the 9 plan fields. |
| `data/interim/` | The 6 clean-step logs (issues, rejected, fixes, dedup). |
| `data/processed/articles_clean.parquet` | Clean data. |
| `data/manifests/` | Split outputs `split_manifest.csv`, `reserve_list.csv`, `annotation_map.csv` (internal). |
| `data/annotations/blind/` | Blind tables `annotation_id,title,category` sent to annotators. |
| `reports/` | `crawl_stats.csv`, `funnel_by_source.csv`, `funnel_by_source_category.csv`, `data_report.md`, `split_report.md`, `split_counts.csv`. |

Flow: `crawl_data.py`/`run_crawling_batch.py` → (`refetch_missing_sapo.py`) →
`export_raw.py` → `build_clean.py` → `build_split.py`. Each step only reads the
previous step's output.

Stage-1 data is frozen: rerun `build_clean.py`/`build_split.py` only when the
config changes (mapping, rules, dedup decisions). To crawl more or rerun
`export_raw.py`, first extract the final runs (the zip stores paths relative to
the repo root):

```powershell
Expand-Archive ../ds-project-archive/crawl_work_final_runs_20261023.zip -DestinationPath .
```

## Confirmed configuration

- 3,500 raw articles per seen outlet, 500 per unseen outlet: 30,000 in total.
- Prefer 2026-01-01 through the end of 2026-10-21, Vietnam time. Only when
  eligible articles run out, backfill from 2025, then 2024, never earlier than
  2024-01-01. Stop once the quota is met. If still short, report it for the
  team to decide. The report gives selected counts per year.
- Build the full inventory per source, drop repeated URLs, shuffle with a fixed
  seed, then fetch. No per-category quota; natural shares are only estimated
  from the inventory found. The category mapping has been approved by the
  team; category-stratified splitting is implemented in
  `scripts/build_split.py` (see Split).
- Any article format is accepted as long as metadata can be extracted. Store
  only the nine plan fields; no body, article HTML, or media.
- For CafeF only, the team allowed treating timestamps without an offset as
  UTC+07:00. The parser records this decision in the journal. Other sources do
  not apply this rule; timestamps missing a timezone must still be reported for
  review.

## Setup and pilot

Python 3.10+, from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/crawl_data.py --sources yan saostar --run-id pilot_metadata_v1 --pilot-per-source 5 --max-discovery-pages 3 --max-attempts 30
```

A pilot may run on an incomplete inventory to test HTTP/parsers; it is not
representative of the category/time distribution. `--max-attempts` caps the
total number of article URLs tried per run/source across resumes, excluding
discovery requests and retries.

## Production crawl and resume

Run all 12 sources, two independent sources at a time:

```powershell
python scripts/run_crawling_batch.py --run-id metadata_2024_20261021 --workers 2 --reuse-discovery-cache
```

Each source keeps its own request delay. The batch starts YAN and SAOstar
first. Status/PID/quota: `data/crawl_work/_batches/<run_id>/status.json`;
per-source stdout: `<source>.console.log` in the same directory. During fetch,
`data/crawl_work/<source>/<run_id>/journal/progress.json` is updated every 25
articles. A source that exits with code 0 is `complete`; otherwise
`needs_review`. Cache reuse only copies discovery XML/metadata from earlier
runs in `data/crawl_work/<source>/`, with provenance in `cache_reuse.json`; it
never mixes articles/selections across runs. A changed config snapshot needs a
new run ID.

Ask a running batch to stop:

```powershell
python scripts/run_crawling_batch.py --run-id metadata_2024_20261021 --stop
```

To run the steps manually:

```powershell
python scripts/crawl_data.py --sources all --stage discover --run-id metadata_20261021
python scripts/crawl_data.py --sources all --stage fetch --run-id metadata_20261021
```

Or `python scripts/crawl_data.py --sources all --run-id metadata_20261021` to
run discovery → fetch source by source. Rerun the same command/run ID to
resume. The script checks the config/profile snapshot; a config change needs a
new run ID. Articles already logged as errors are not retried automatically;
after fixing a parser/network issue, use a new run to keep the original
results.

Defaults: serial, at least one second between requests, timeouts, and limited
retries for network errors/429/5xx. 403s are never retried in bulk. Settings
live in `configs/crawling/run.json`; the source list and plan clean quotas in
`sources.csv`.

If `--max-discovery-pages` (default 2,000 per source) is hit or an endpoint
fails, the production fetch stops to avoid sampling from a partial inventory.
Read the discovery report, fix the error or raise the budget, then rerun
discovery; the cache avoids re-downloading what was already found. Each run
holds a `.running` lock; after a hard kill, delete the lock only once you have
verified that process has stopped. Never run two processes writing the same
source/run ID.

SAOstar uses the monthly sitemaps directly because the 2026 year index
redirects to the home page. YAN uses the public `ContentList/TinMoiLoadMore`
endpoint found in the site's JavaScript, walking "Tin mới" and every category
linked from the menu; "Tin mới" alone can yield too few URLs. The YAN profile
uses a browser-compatible User-Agent; no account is needed.

### Run working directory

Inside `data/crawl_work/<source>/<run_id>/`:

- `articles.jsonl`: exactly the nine fields, including failed fetch/parse attempts.
- `discovered_urls.jsonl`, `discovery_cache/`, `discovery_report.json`:
  inventory, URL provenance, parsed XML/metadata cache, and coverage limits.
  HTML is not cached.
- `run_manifest.json`: config snapshot for resume; `cache_reuse.json` when a
  cache was reused.
- `selected_record_ids.csv`: IDs counted toward the raw target, marked primary
  or fallback.
- `crawl_report.json`: eligible counts, quota shortfall, and reasons for not
  selecting.
- `journal/`: `requests.jsonl`, `article_attempts.jsonl` (resume, selector
  evidence, final URL after redirects, errors), `robots.txt`, `progress.json`,
  and `failure.json` if the source did not complete.
- `sapo_refetch/<refetch_id>/`: sapo refetch results (see below).

The ID is the SHA-256 of source + discovery URL with the fragment removed;
query strings are not stripped. `url` keeps the requested URL; the final
redirect target is in the journal. An article counts toward the quota only if
it has a title, metadata confirming it is an article page, a determinable
publish date within the allowed window, and no replacement characters in
title/sapo/category. Missing sapo/category is kept and reported as `partial`;
the SEO description or body is never used as a stand-in sapo.

A fully walked inventory only means every endpoint found was walked; it **does
not prove coverage of the outlet's full history**. Every shortfall/failing
endpoint is reported explicitly. Meeting the raw quota does not prove the clean
quota is met after dedup.

VietNamNet/Nhân Dân enable broken-XML recovery after cross-checking against
the old crawler. The recovering parser must account for every record and
match each `loc` from the source with a strict parser, never using URLs the
recovering parser rewrote. Details are in `xml_recoveries` of the discovery
report. XML that is truncated, structurally broken, or has unverifiable URLs
still marks the inventory incomplete. Quarantining broken URLs is a separate
option, `quarantine_invalid_sitemap_urls`, off by default. On 2026-10-23 the
team enabled it for VietNamNet only: 13 records (11 URLs from 2022) with
control characters in `loc` were quarantined, URLs not rewritten; the list is
in `xml_recoveries[].quarantined_urls` of run `metadata_fix_20261023`.

Discovery skips `sitemap-tags-NN.xml` sitemaps and `-tagNNN.html` tag-page URLs
(2sao, VietNamNet): tag pages match the article-ID pattern but are not
articles.

Dân Trí allows roughly 100 requests per 3 minutes and returns HTTP 428. Its
profile sets `request_delay_seconds` 2.5 and `rate_limit_cooldown_seconds` 120
(the wait before retrying a 428; a longer `Retry-After` takes precedence). Two
Dân Trí formats are systematically excluded by team decision: "Dân trí 360"
(short videos rendered in JavaScript, no title/date in the HTML) and
emagazines whose headline is an image; both currently end up as
`missing_title`. YAN has no separate sapo: only the first paragraph of
`#contentBody` is accepted, and only if it is entirely bold
(`sapo_require_full_bold_paragraph`).

### Refetching sapo for selected articles

After fixing a sapo selector there is no need to recrawl the whole source:
refetch only the selected articles missing a sapo and write a supplement next
to the run; the run's `articles.jsonl` is unchanged. A sapo is accepted only
when the refetched title equals the stored title. Rerun the same command to
resume; the script stops itself after many consecutive errors (e.g. 428).

```powershell
python scripts/refetch_missing_sapo.py --source dantri --run-id metadata_2024_20261021 --refetch-id sapo_refetch_dantri_20261023 --request-delay 4
```

Output: `data/crawl_work/<source>/<run_id>/sapo_refetch/<refetch_id>/` with
`sapo_supplement.jsonl` (`sapo_recovered`/`sapo_absent`/`title_changed`),
`manifest.json`, `report.json`, and `journal/` (request log). Declare it under
`sapo_supplements` in `configs/crawling/final_runs.json` so `export_raw.py`
fills it into raw.

## Export raw

`configs/crawling/final_runs.json` lists the final run per source (`runs`) and
the sapo refetches (`sapo_supplements`).

```powershell
python scripts/export_raw.py            # first run
python scripts/export_raw.py --replace  # overwrite after changing final_runs.json
```

The script only reads `data/crawl_work/` and writes:

- `data/raw/articles_raw.jsonl`: UTF-8, one object per line with exactly 9
  fields (`record_id, source, url, title, sapo, category, published_at,
  crawled_at, extraction_status`). Order: sources as in `sources.csv`, then
  `selected_record_ids.csv` order within each source.
- `reports/crawl_stats.csv`: one row per source: run, inventory, attempts,
  `downloaded`/`downloaded_without_title`, selected/target/shortfall, attempt
  counts per `selection_status` (`selection_*`), publish years,
  `success`/`partial`, `missing_<field>` on the exported raw,
  `missing_sapo_before_refetch`, `sapo_refetched`, number of raw categories,
  date range, and `downloaded_by_category` (JSON
  `[[raw category, count], ...]`, used for the per-category funnel).

**Refetched sapo:** fills only rows without a sapo, using only
`sapo_recovered` rows. After filling, the row's `extraction_status` is
recomputed with the crawler's rule: `success` when title, sapo, category and
published_at are all present; otherwise `partial`. Hence the 310 Dân Trí
articles with a recovered sapo are `success` in raw. Raw has no separate flag
for them; their record_ids are in the refetch's `sapo_supplement.jsonl`
(status `sapo_recovered`), inside `crawl_work_final_runs_20261023.zip`.

The script refuses (writes nothing) when a run: does not exist, still holds a
`.running` lock, lacks `selected_record_ids.csv`/`crawl_report.json`/
`discovery_report.json`, has `shortfall > 0`, is a pilot run
(`--pilot-per-source`), selected fewer than the source group's `raw_targets` in
`configs/crawling/run.json`, has a selected ID with no row (or several rows) in
`articles.jsonl`, or has malformed JSONL; or when a declared sapo refetch has
no `report.json` or has `remaining > 0`. Existing output requires `--replace`.

## Clean: checking, category mapping, dedup

```powershell
python scripts/build_clean.py            # first run
python scripts/build_clean.py --replace  # overwrite previous output
```

Input: `data/raw/articles_raw.jsonl` and `reports/crawl_stats.csv` (if missing,
it errors; run `export_raw.py` first). Per-source article counts in raw must
match the `selected` column of `crawl_stats.csv`. Output (existing files
require `--replace`):

| Location | Files |
| --- | --- |
| `data/interim/` | `issues.csv` (record_id, issue, decision reject/flag/report, reason), `rejected.csv`, `fixes.csv` (before/after), `dedup_pairs.csv`, `dedup_review.csv`, `template_series.csv` |
| `data/processed/` | `articles_clean.parquet` (`source_run_id` taken from the `run_id` in `crawl_stats.csv`) |
| `reports/` | `funnel_by_source.csv`, `funnel_by_source_category.csv`, `data_report.md` |

`data_report.md` is the single report, in two parts: **Raw crawl** (table from
`crawl_stats.csv`, selection/non-selection reasons, policies applied) and
**Clean** (funnel, rejected articles, issue/technical-fix counts, dedup,
categories, limitations).

Funnel definitions: **downloaded** = distinct `record_id`s across the crawl
run's entire `articles.jsonl` with `extraction_status` `success`/`partial`
**and a title** (including out-of-window/unselected articles); rows without a
title (tag/listing pages, missing_title) are reported separately as
`downloaded_without_title`; **valid** = selected articles that pass checking;
**after dedup** = non-excluded group representatives; **with sapo** is counted
within after dedup. `shortfall` is measured against `min_clean_articles` in
`sources.csv`.

Config in `configs/checking/`: `rejection_rules.json` (rules rejecting titles
that are not article headlines: Nhân Dân "Tin tức"/print-edition pages, Thanh
Niên print-edition items, lottery/Vietlott results, correction notices; every
match is logged), `dedup.json` (JC3 thresholds, number-blocking type,
`near_link_max_days` = 7: near-duplicate pairs more than 7 days apart are not
merged automatically but go to review as `date_gap`), and
`dedup_decisions.csv` (`link`/`separate` after review). A link is rejected
(`transitive_conflict`) if it would put a number-mismatched pair or a
`separate` pair into the same group via a third article.

After changing a final run, add mapping proposals for new categories (existing
rows are not overwritten), then review them manually before running clean.
`--all-rows` uses the categories of every downloaded article (from
`crawl_stats.csv`) instead of only those in raw:

```powershell
python scripts/propose_category_mapping.py --dry-run
python scripts/propose_category_mapping.py
```

## Split: manifest, reserve, blind tables

```powershell
python scripts/build_split.py            # first run
python scripts/build_split.py --replace  # overwrite previous output
```

Input (read-only): `data/processed/articles_clean.parquet`,
`data/interim/{template_series,dedup_review,dedup_pairs}.csv` (if missing, run
`build_clean.py` first), `configs/checking/dedup_decisions.csv`, quotas from
`configs/crawling/sources.csv`, seed from `configs/crawling/run.json` (2026).
Output (existing files require `--replace`):

| Location | Files |
| --- | --- |
| `data/processed/` | `dataset.csv`: combined table of the 23,600 split articles (`record_id, annotation_id, split, is_human_subset, source, outlet_group, category, title, sapo, published_at, url, dup_group`; `annotation_id` is set for the 10,000 human-scope articles) |
| `data/manifests/` | `split_manifest.csv` (23,600 rows), `reserve_list.csv`, `annotation_map.csv` (10,000 rows, internal) |
| `data/annotations/blind/` | `blind_dev.csv` (1,400 rows, labelled first), `blind_train_subset_test.csv` (8,600 rows); only `annotation_id,title,category` |
| `reports/` | `split_report.md`, `split_counts.csv` (source, category, split, records) |

Rules:

- Pool = duplicate-group representatives with no `dedup_excluded_reason`. The
  script refuses to run if a pool article has a non-null category whose
  mapping is not `approved`. Articles with a null category are never selected;
  they go to reserve under the stratum `(không có category)`.
- The sampling unit is the `split_group`: `dup_group`s connected by a
  near-duplicate pair that clean lists but does not merge
  (`template_series.csv`: same-source templated title series;
  `dedup_review.csv`: pairs awaiting review; `dedup_pairs.csv`: blocked links)
  are combined so two near-identical titles never land in different splits;
  the articles are kept, not deleted. A seen–unseen pair cannot share a split:
  unless `dedup_decisions.csv` has a `separate` decision, the seen article (and
  its whole `split_group`) is not selected and goes to reserve with
  `hold_reason=pending_dedup_review_with_unseen`. Review these pairs before
  labelling: changing a decision and rerunning clean + split changes the
  split. Quotas count articles: a group that does not fit the remaining quota
  is skipped and the next group is tried; if any cell cannot be filled
  exactly, the script stops and reports the numbers.
- Per seen source: train 2,500 (of which human subset 800) + dev 175 +
  test_seen 125; per unseen source: test_unseen 300. Each split quota
  (including the subset and test_unseen) is allocated by the pool's category
  shares within the source using largest remainder (ties broken by category
  name). Labels are not used.
- Dev takes only articles with a sapo (plan §3.1). Train/subset/test do not
  require a sapo.
- Order: per (source, category), shuffle the `split_group`s with
  `random.Random("<seed>:<source>:<category>")`; fill dev → test_seen → train
  (unseen: test_unseen), then pick the subset within train with its own seed
  (`...:human_subset`). A group spanning several cells is considered at its
  earliest relative position, and all cells advance in lockstep. Remaining
  articles go to reserve, with `reserve_rank` continuing the seeded order
  within (source, category); it is used to replace articles dropped during
  labelling (plan §2.4): take the smallest `reserve_rank` in the same
  source/category with an empty `hold_reason`, taking the whole
  `split_group`; a dev replacement must have `has_sapo=True`.
- `annotation_id` = `A` + 5 digits drawn at random with a seed from
  10000–99999; blind-table rows follow a seeded permutation that mixes
  sources/categories/splits.
- Checks (the script stops on failure): exact quota per source × split and
  subset; each `split_group`/`dup_group` sits in exactly one split and is not
  split off into reserve; no train/dev/test_seen article shares a group with
  an unseen article; no unmerged near-duplicate pair (except `separate` pairs)
  spans two splits; articles held for review are not in any split;
  subset ⊂ train; dev all has sapo; the blind/map tables cover exactly 10,000
  articles; building twice in the same command yields byte-identical output.
  Reports do not record the run time, so rerunning with the same input yields
  identical files.

`annotation_assignment.csv` (plan §2.2: chunks C1..C9, people P1..P9) is not
produced by this script; the human annotation lead builds it from
`annotation_map.csv` and the blind tables.

## Tests

```powershell
python -m unittest discover -s tests -v
```
