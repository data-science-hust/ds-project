# DS Project 2026.1 — Vietnamese Clickbait Detection (Group 5)

Clickbait headline detection for Vietnamese online news.

## Research questions

- **RQ1:** How closely do LLM-generated clickbait labels agree with independent
  human annotations, and how does the input context (headline only,
  headline + topic, headline + sapo, headline + sapo + topic) affect that agreement?
- **RQ2:** How well do classifiers trained on the LLM-labelled, partially
  human-corrected dataset perform on seen and unseen outlets, and how does
  progressively correcting LLM labels on a fixed training subset affect them?

## Project flow

| # | Stage | Content |
| --- | --- | --- |
| 1 | Crawl & data preparation | Crawl 12 outlets (8 seen, 4 unseen); check, deduplicate, map categories; split train/dev/test and build blind tables |
| 2 | Human annotation | 10,000 titles (train subset + dev + both tests), two independent annotators per title plus adjudication |
| 3 | LLM annotation | Try LLMs × 4 input variants on dev, compare with human gold (EVAL1), freeze the best configuration and label the full train set |
| 4 | Label versions | Full train set after correction (human gold on the subset, LLM labels elsewhere) and nested correction levels 0–100% on the subset |
| 5 | EDA | Distributions by split / source / category / label, LLM–human disagreement |
| 6 | Preprocessing | TF-IDF for ML models, word segmentation + PhoW2V for RNN/CNN, model tokenizers for PhoBERT / XLM-R |
| 7 | Model training | 9 models (LR, Linear SVM, RF, XGBoost, PhoBERT-large, XLM-R-large, BiLSTM, BiGRU, TextCNN); correction-scaling ablation with LR and PhoBERT |
| 8 | EVAL2 | Macro-F1 on test seen and test unseen, paired bootstrap, error analysis |
| 9 | Report | |

Timeline: **Phase 1** full data crawl → **Phase 2** annotation (dev first, then
train subset + tests) and preprocessing → **Phase 3** label versions and
training → **Phase 4** EVAL2 → **Phase 5** report.

This repository currently implements **stage 1** (complete); its outputs are
the input for stages 2–3.

## Stage 1 results

| Step | Articles |
| --- | ---: |
| Raw crawl: 3,500 per seen outlet × 8, 500 per unseen outlet × 4, all published in 2026 | 30,000 |
| Valid after checks (64 non-news titles + 159 Eva articles without sapo removed) | 29,777 |
| After cross-source exact and near-duplicate removal | 29,434 |
| **Split** | **23,600** |
| — train (incl. 6,400 human-labelled subset) | 20,000 |
| — dev (100% with sapo) | 1,400 |
| — test seen | 1,000 |
| — test unseen (saostar, yan, vietnamnet, nhandan) | 1,200 |
| Reserve for replacing faulty records during annotation | 5,834 |
| Human annotation scope (subset + dev + both tests) | 10,000 |

Seen outlets: afamily, 2sao, eva, kenh14, cafef, thanhnien, tuoitre, dantri.
Raw outlet categories are mapped to 16 approved common groups; within each
seen outlet, train/dev/test share the same category proportions. Details:
`reports/data_report.md`, `reports/split_report.md`,
`reports/category_mapping_review.md`.

## Repository layout

`src/ds_project/` holds the logic (imported by scripts and tests); `scripts/`
holds one command per pipeline step (arguments, file reading/writing).

```text
ds-project/
├── configs/
│   ├── crawling/
│   │   ├── sources.csv                  # 12 outlets: seen/unseen group, raw quota
│   │   ├── run.json                     # date range, seed (2026)
│   │   ├── profiles.json                # per-outlet discovery + CSS selectors, rate limits
│   │   ├── final_runs.json              # final crawl run per outlet (input of export_raw)
│   │   └── category_mapping.csv         # (source, raw category) → 16 common groups, all approved
│   └── checking/
│       ├── rejection_rules.json         # record rejection rules (non-news titles, Eva without sapo, ...)
│       ├── dedup.json                   # near-duplicate thresholds
│       └── dedup_decisions.csv          # manual merge/separate decisions on near-duplicate pairs
├── src/ds_project/
│   ├── crawling/
│   │   ├── discover.py                  # sitemap/feed discovery → URL inventory
│   │   ├── network.py                   # rate-limited requests, retry on 428/429/5xx, request log
│   │   └── extract.py                   # title, sapo, category, date from the article page
│   ├── checking/
│   │   ├── clean.py                     # checks, technical fixes, category join → clean table
│   │   ├── dedup.py                     # exact + near-duplicate grouping (dup_group)
│   │   ├── categories.py                # 16 common groups, mapping lookup and proposals
│   │   └── report.py                    # reports/data_report.md
│   └── splitting/
│       ├── split.py                     # stratified split, reserve list, blind tables, checks
│       └── report.py                    # reports/split_report.md, split_counts.csv
├── scripts/
│   ├── crawl_data.py                    # crawl one outlet (inventory, resume)
│   ├── run_crawling_batch.py            # crawl all outlets to quota, in parallel
│   ├── refetch_missing_sapo.py          # re-fetch articles whose sapo is missing
│   ├── export_raw.py                    # final runs → data/raw/articles_raw.jsonl
│   ├── propose_category_mapping.py      # add new raw categories to category_mapping.csv
│   ├── build_clean.py                   # raw → clean + dedup + funnel reports
│   └── build_split.py                   # clean → split, dataset.csv, blind tables
├── tests/                               # unittest: test_crawling, test_clean, test_split
├── docs/
│   ├── pipeline.md                      # how to run each step, rules in detail
│   └── data_contract.md                 # schema of every output file
├── data/
│   ├── raw/articles_raw.jsonl           # 30,000 crawled articles, 9 plan fields
│   ├── interim/
│   │   ├── rejected.csv                 # 223 rejected records with reasons
│   │   ├── issues.csv                   # issues found (missing sapo, rejects, flags)
│   │   ├── fixes.csv                    # technical fixes (before → after)
│   │   ├── dedup_pairs.csv              # duplicate links, applied or blocked
│   │   ├── dedup_review.csv             # near-duplicate pairs for review
│   │   └── template_series.csv          # same-template title series (kept in one split)
│   ├── processed/
│   │   ├── articles_clean.parquet       # 29,777 valid records with dup_group, category
│   │   └── dataset.csv                  # ★ 23,600 split records, all fields + annotation_id
│   ├── manifests/
│   │   ├── split_manifest.csv           # record_id → split, is_human_subset, split_group
│   │   ├── annotation_map.csv           # annotation_id → record_id (internal)
│   │   └── reserve_list.csv             # 5,834 replacement records, ranked
│   └── annotations/blind/
│       ├── blind_dev.csv                # 1,400 rows: annotation_id, title, category
│       └── blind_train_subset_test.csv  # 8,600 rows: annotation_id, title, category
└── reports/
    ├── crawl_stats.csv                  # crawl counts per outlet
    ├── data_report.md                   # raw → valid → after dedup → with sapo
    ├── funnel_by_source*.csv            # same funnel per outlet / per category
    ├── category_mapping_review.md       # mapping review notes
    ├── split_report.md                  # split counts, category balance, checks
    └── split_counts.csv                 # records per source × category × split
```

## Setup

Python 3.10+:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Running the pipeline

| Step | Command | Main output |
| --- | --- | --- |
| 1. Crawl | `python scripts/run_crawling_batch.py --run-id <id> --workers 2` | `data/crawl_work/<source>/<run_id>/` |
| (optional) Re-fetch sapo | `python scripts/refetch_missing_sapo.py --source <s> --run-id <id> --refetch-id <id>` | `.../sapo_refetch/<id>/` |
| 2. Export raw | `python scripts/export_raw.py` | `data/raw/articles_raw.jsonl`, `reports/crawl_stats.csv` |
| 3. Clean + dedup | `python scripts/build_clean.py` | `data/processed/articles_clean.parquet`, `data/interim/*.csv`, `reports/data_report.md` |
| 4. Split | `python scripts/build_split.py` | `data/processed/dataset.csv`, `data/manifests/*.csv`, `data/annotations/blind/*.csv`, `reports/split_report.md` |

Add `--replace` to overwrite existing outputs. The data is frozen, so usually
only steps 3–4 are re-run after a config change. The crawler working directory
and crawl logs are stored as zip files outside the repository
(`../ds-project-archive/`, only on the crawling machine); unzip them before
re-running steps 1–2. Full guide: [docs/pipeline.md](docs/pipeline.md).

## Stage 1 deliverables

| Plan item | File |
| --- | --- |
| **Final dataset: the 23,600 split records with all fields** | `data/processed/dataset.csv` |
| Clean data with `dup_group` (all 29,777 valid records) | `data/processed/articles_clean.parquet` |
| Category mapping | `configs/crawling/category_mapping.csv` |
| Count report (downloaded → valid → after dedup → with sapo) | `reports/data_report.md`, `reports/funnel_by_source*.csv` |
| Split manifest (`record_id, source, split, dup_group, is_human_subset, seed`) | `data/manifests/split_manifest.csv` |
| Blind tables (`annotation_id, title, category`) | `data/annotations/blind/blind_dev.csv` (1,400), `blind_train_subset_test.csv` (8,600) |
| Internal annotation mapping (not sent to annotators) | `data/manifests/annotation_map.csv` |
| Reserve list | `data/manifests/reserve_list.csv` |

## Inputs for the next stages

**`data/processed/dataset.csv` is the one file to use.** It holds the 23,600
split records, one row each, with columns `record_id, annotation_id, split,
is_human_subset, source, outlet_group, category, title, sapo, published_at,
url, dup_group`. `annotation_id` is filled for the 10,000 human-annotation
records and empty for the rest of train; it links human labels from the
annotation tool back to the dataset. Splits: `train` (20,000, of which
`is_human_subset=True` 6,400), `dev` (1,400), `test_seen` (1,000),
`test_unseen` (1,200). Row order carries no meaning; join on `record_id` or
`annotation_id`.

### Human annotation (stage 2)

| What | File |
| --- | --- |
| Round 1 — dev, labelled first | `data/annotations/blind/blind_dev.csv` (1,400 rows) |
| Round 2 — train subset + test seen + test unseen | `data/annotations/blind/blind_train_subset_test.csv` (8,600 rows) |
| Map labels back (`annotation_id` → record) | `annotation_id` column of `data/processed/dataset.csv` |
| Replacements for faulty titles | `data/manifests/reserve_list.csv` |

Blind tables contain only `annotation_id, title, category` (no source, URL or split). To replace a faulty record,
take the lowest `reserve_rank` with the same `source` and `category` and an
empty `hold_reason` (whole `split_group`); a dev replacement must have
`has_sapo=True`. The chunk/annotator assignment (`annotation_assignment.csv`)
is built by the annotation lead.

### LLM annotation (stage 3)

Take everything from `data/processed/dataset.csv`; no other file is needed.

```python
import pandas as pd

data = pd.read_csv("data/processed/dataset.csv", encoding="utf-8-sig")
dev = data[data.split == "dev"]      # 1,400: EVAL1, the four input variants
train = data[data.split == "train"]  # 20,000: full LLM labelling with the frozen configuration
```

| Input variant (plan 3.1) | Columns |
| --- | --- |
| Headline | `title` |
| Headline + topic | `title`, `category` |
| Headline + sapo | `title`, `sapo` |
| Headline + sapo + topic | `title`, `sapo`, `category` |

All 1,400 dev records have both `sapo` and `category`, so the four variants
run on the same records.
