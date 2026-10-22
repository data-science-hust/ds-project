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

```text
configs/
  crawling/   sources.csv (outlets, quotas), run.json (date range, seed), profiles.json (per-outlet selectors),
              final_runs.json (final run per outlet), category_mapping.csv (raw category → common group)
  checking/   rejection_rules.json, dedup.json, dedup_decisions.csv
src/ds_project/
  crawling/   sitemap/feed discovery, page fetching, extraction of the 9 plan fields
  checking/   checks, technical fixes, category mapping, deduplication, clean report
  splitting/  split, reserve list, blind tables, split report
scripts/      command-line entry point for each step (see below)
tests/        unittest suites
docs/         pipeline.md (how to run + rules), data_contract.md (schemas)
data/         raw/, interim/, processed/, manifests/, annotations/blind/
reports/      generated reports
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
| 4. Split | `python scripts/build_split.py` | `data/manifests/*.csv`, `data/annotations/blind/*.csv`, `reports/split_report.md` |

Add `--replace` to overwrite existing outputs. The data is frozen, so usually
only steps 3–4 are re-run after a config change. The crawler working directory
and crawl logs are stored as zip files outside the repository
(`../ds-project-archive/`, only on the crawling machine); unzip them before
re-running steps 1–2. Full guide: [docs/pipeline.md](docs/pipeline.md).

## Stage 1 deliverables

| Plan item | File |
| --- | --- |
| Clean data with `dup_group` | `data/processed/articles_clean.parquet` |
| Category mapping | `configs/crawling/category_mapping.csv` |
| Count report (downloaded → valid → after dedup → with sapo) | `reports/data_report.md`, `reports/funnel_by_source*.csv` |
| Split manifest (`record_id, source, split, dup_group, is_human_subset, seed`) | `data/manifests/split_manifest.csv` |
| Blind tables (`annotation_id, title`) | `data/annotations/blind/blind_dev.csv` (1,400), `blind_train_subset_test.csv` (8,600) |
| Internal annotation mapping (not sent to annotators) | `data/manifests/annotation_map.csv` |
| Reserve list | `data/manifests/reserve_list.csv` |
