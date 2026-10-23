# Data contract for crawling/checking

The first nine fields below follow plan §1.1. The ID convention, file formats,
timestamps, and status values are this repo's implementation choices.

As confirmed by the team, the crawler writes exactly the nine §1.1 fields to
`articles.jsonl`. The delivered raw dataset is `data/raw/articles_raw.jsonl`
(`scripts/export_raw.py`): exactly these nine fields, ordered by source as in
`sources.csv`, then by selection order in `selected_record_ids.csv`. The three
fields `category_raw`, `title_raw`, `dup_group` below are the contract for the
clean/dedup step and are never added to crawl output. Evidence/parser/resume
data lives separately in the journal
(`data/crawl_work/<source>/<run_id>/journal/`); no HTML/body is stored.
Team-approved timestamp exception: CafeF timestamps without an offset use
UTC+07:00. This is configured for that source only and the evidence is logged.

| Field | Type | Convention |
| --- | --- | --- |
| `record_id` | string | Stable article ID, never a dataframe index. Proposed: hash of source + canonical URL; canonicalization rule fixed before the full crawl. |
| `source` | string | Slug from `configs/crawling/sources.csv`; never used as a clickbait label. |
| `url` | string/null | Source URL; must be non-empty to enter clean. Raw keeps the original URL; a canonical URL may be stored separately. |
| `title` | string/null | Original headline; must be non-empty/non-whitespace to enter clean. Writing style is never altered. |
| `sapo` | string/null | Lead paragraph; null when missing, never filled from the full body or the title. |
| `category` | string/null | Raw: the outlet's category name. Clean: the mapped shared category. Unmapped values must be reported. |
| `published_at` | string/null | ISO 8601 with offset when determinable; dates and timezones are never invented. |
| `crawled_at` | string | ISO 8601 with offset; actual fetch time. |
| `extraction_status` | string | `success` (title, sapo, category, published_at all present), `partial`, `fetch_error`, `parse_error`; not a data-checking verdict. Exported raw contains only `success`/`partial`; rows filled with a refetched sapo have their status recomputed with the same rule. |
| `category_raw` | string/null | Original category kept in clean to trace the mapping. |
| `title_raw` | string/null | Title before clean's technical fixes; raw stays unchanged. |
| `dup_group` | string | Required in clean before split; singleton articles get their own group. |

## `articles_clean.parquet` columns

Each row is a **valid** article (passed checking), including
non-representatives of duplicate groups so split can sample by group. Rejected
articles are in `data/interim/rejected.csv`; raw is unchanged.

| Column | Type | Convention |
| --- | --- | --- |
| `record_id`, `source`, `url` | string | As in raw; `record_id` is unique. |
| `outlet_group` | string | `seen`/`unseen` per `sources.csv`. |
| `title` | string | `title_raw` with leading/trailing whitespace stripped + NFC; letters, case and punctuation unchanged. |
| `title_raw` | string | Original title string from raw. |
| `sapo` | string/null | Strip + NFC; empty → null. |
| `category` | string/null | Shared group from `category_mapping.csv`; null when unmapped. |
| `category_raw` | string/null | Outlet category after `html.unescape` + NFC + strip; empty → null. The mapping join key. |
| `category_mapping_status` | string/null | The mapping row's `review_status` (`proposed`/`needs_review`/`approved`); null when `category` is null (missing key or mapping row without a category). Split only accepts articles with a non-null `category` **and** status `approved`. |
| `published_at`, `crawled_at`, `extraction_status` | string | As in raw. |
| `dup_group` | string | `dg_` + first 12 hex chars of the SHA-256 of the sorted member `record_id`s; singletons get a group too. |
| `dup_group_size` | int | Number of valid articles in the group. |
| `is_dup_representative` | bool | The article kept for the group. |
| `dedup_excluded_reason` | string/null | `duplicate_in_group` or `duplicate_of_unseen` (seen copy of an unseen article, excluded from every seen split); null when kept. |
| `has_sapo` | bool | Has a sapo after technical fixes. |
| `source_run_id` | string | Crawl run that produced the article: from `configs/crawling/final_runs.json`, recorded via the `run_id` column of `reports/crawl_stats.csv`. |

Representative: a group containing unseen articles keeps one unseen copy
(earliest `published_at`, then `record_id`); a seen-only group keeps the
earliest article, then `record_id`. "After dedup" = `is_dup_representative`
with no `dedup_excluded_reason`.

Raw JSONL is UTF-8, one object per line. In the crawler's working directory,
records that could not be extracted are still stored with their status. The
request log keeps HTTP status/errors and references the raw record. A
successful fetch is distinct from a valid record: `success` does not guarantee
passing checking.

## Checking

- Required fields, data types, duplicate IDs, missing URL/title,
  whitespace-only titles.
- Invalid URLs or URLs outside the expected source; unparseable timestamps.
- Broken encoding or suspected mojibake: sent to review, fixed only with
  evidence.
- Duplicate URL/canonical URL; duplicate titles and cross-source near
  duplicates. Normalization for comparison uses auxiliary fields and never
  overwrites the displayed title.
- Missing sapo; missing/unmapped category; statistics per source/category.
- Issue log per `record_id` with issue type, decision and reason. Records not
  eligible for clean are still kept in raw and in the rejected table.
- Before freeze: every clean record has a unique ID, valid title/URL, a
  `dup_group`, and a handled category; quotas are met after dedup and leakage
  removal.

All of the above is implemented in `scripts/build_clean.py`; rules rejecting
titles that are not article headlines are in
`configs/checking/rejection_rules.json`. Mojibake is rejected only when a
cp1252/latin-1 → UTF-8 round trip of the title confirms it; otherwise it is
flagged for review. Every mapping row is `approved` (16 groups); split refuses
to run if any used row is not approved. An `approved` mapping row must have a
category. Dedup: a near-duplicate link requires publish dates ≤ 7 days apart
and must not transitively merge two number-mismatched articles; see
`dedup_review.csv`.

## Handoff tables after checking

The split tables are produced by `scripts/build_split.py` (UTF-8 with BOM,
standard CSV; titles containing newlines are quoted). Rerunning with the same
input and seed gives byte-identical files.

`data/manifests/split_manifest.csv`: the first six columns follow the plan's
order, the last three are repo additions; one row per selected article (23,600
rows).

```csv
record_id,source,split,dup_group,is_human_subset,seed,split_group,category,outlet_group
```

| Column | Convention |
| --- | --- |
| `split` | `train`, `dev`, `test_seen`, `test_unseen`. |
| `is_human_subset` | `True`/`False`; `True` only for the 6,400-article subset **within train**. Dev/test are also in human-annotation scope; this flag is not the full human scope. |
| `seed` | Seed from `configs/crawling/run.json` (2026). |
| `split_group` | Sampling unit: `sg_` + first 12 hex chars of the SHA-256 of the member `dup_group`s (sorted, newline-joined). Combines `dup_group`s connected by an unmerged near-duplicate pair on the same seen/unseen side (`data/interim/template_series.csv`, `dedup_review.csv`, blocked links in `dedup_pairs.csv`); an unconnected group has a single `dup_group`. |
| `category`, `outlet_group` | As in `articles_clean.parquet` (category is always non-null and `approved`). |

No `dup_group` or `split_group` appears in more than one split or is divided
between a split and reserve.

`data/processed/dataset.csv`: combined table of the 23,600 split articles
(manifest rows joined with clean fields); `annotation_id` is set for the 10,000
human-scope articles and empty otherwise.

```csv
record_id,annotation_id,split,is_human_subset,source,outlet_group,category,title,sapo,published_at,url,dup_group
```

`data/manifests/reserve_list.csv`: every after-dedup article not selected.

```csv
record_id,source,category,dup_group,split_group,reserve_rank,has_sapo,hold_reason
```

`category` = `(không có category)` for null-category articles (never
selected). `reserve_rank` starts at 1 within each (`source`, `category`),
continuing the split step's seeded order. `has_sapo` (`True`/`False`) as in the
clean table. `hold_reason` empty = usable; `no_category` = the group has a
null-category article; `pending_dedup_review_with_unseen` = the group has a
seen article near-duplicating an unseen one with no dedup decision yet (not
selectable until reviewed). Replacement: the smallest `reserve_rank` in the
same source/category with an empty `hold_reason`, taking the whole
`split_group` if it has several articles; **a dev replacement needs
`has_sapo=True` for the whole group** (articles without sapo are skipped).

Blind tables in `data/annotations/blind/`: `blind_dev.csv` (1,400 dev
articles, labelled first) and `blind_train_subset_test.csv` (8,600 = subset
6,400 + test_seen 1,000 + test_unseen 1,200), three columns only:

```csv
annotation_id,title,category
```

`annotation_id` = `A` + 5 digits (drawn at random with a seed from
10000–99999) and reveals nothing about source, split or `record_id`; row order
is a seeded permutation. `title` is the clean title (no further edits);
`category` is the approved shared group (16 groups).

`data/manifests/annotation_map.csv` (internal, never sent to annotators),
10,000 rows sorted by `annotation_id`:

```csv
annotation_id,record_id,split,source
```

`reports/split_counts.csv`: `source,category,split,records`, where `split` is
one of `train`, `train_human_subset` (subset of train), `dev`, `test_seen`,
`test_unseen`, `reserve`.

`category_mapping.csv`:

```csv
source,category_raw,category,review_status,notes
```

The mapping key is `(source, category_raw)` in normalized form
(`html.unescape` + NFC + strip); each key maps to one shared category from the
16 groups in `src/ds_project/checking/categories.py`. `review_status` is
`proposed`, `needs_review` or `approved` (set only by the team); an empty
category means unmapped. `review_status` and `notes` are repo-added metadata.

The annotation ID → record ID map and `annotation_assignment.csv` live in the
internal manifests and are never attached to the blind tables.
`annotation_assignment.csv` (plan §2.2: chunks C1..C9, people P1..P9) is built
by the human annotation lead; its schema is fixed when chunks are assigned, and
the split step does not create it.
