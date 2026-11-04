"""Run Exploratory Data Analysis (EDA) on the dataset and annotations.

Generates:
- reports/eda_report.md
- reports/eda_summary_by_source.csv
- reports/eda_summary_by_category.csv
- reports/eda_disagreement.csv
- reports/eda_label_distribution.csv

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.eda.analysis import (
    analyze_by_category,
    analyze_by_source,
    analyze_human_annotations,
    analyze_llm_variants,
    analyze_splits_and_sources,
    analyze_train_labels,
    compute_text_stats,
)
from ds_project.eda.report import render_eda_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset EDA and generate report.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/dataset.csv")
    parser.add_argument("--dev-gold", type=Path, default=ROOT / "data/annotations/human/gold/dev_gold.csv")
    parser.add_argument("--adjudication-log", type=Path, default=ROOT / "data/annotations/human/dev/adjudication_log.csv")
    parser.add_argument("--llm-dir", type=Path, default=ROOT / "data/annotations/llm")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/processed/train_full_corrected.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    df_dataset = pd.read_csv(args.dataset)

    # 1. Split, source, and category analyses
    df_splits_sources = analyze_splits_and_sources(df_dataset)
    df_by_category = analyze_by_category(df_dataset)
    df_by_source = analyze_by_source(df_dataset)

    # 2. Headline text stats
    text_stats = compute_text_stats(df_dataset["title"])

    # 3. Human annotation agreement on dev
    human_stats = analyze_human_annotations(
        dev_gold_path=args.dev_gold,
        adjudication_log_path=args.adjudication_log,
    )

    # 4. LLM variants evaluation & disagreement on dev
    df_llm_variants, df_disagreement = analyze_llm_variants(
        dev_gold_path=args.dev_gold,
        llm_dir=args.llm_dir,
        df_dataset=df_dataset,
    )

    # 5. Train set label distributions
    if args.train_data.exists():
        df_train = pd.read_csv(args.train_data)
    else:
        df_llm_train = pd.read_csv(args.llm_dir / "llm_train.csv", low_memory=False)
        df_train = df_dataset[df_dataset["split"] == "train"].copy()
        llm_map = df_llm_train.set_index("record_id")["label"].to_dict()
        df_train["label"] = [llm_map.get(r, 0) for r in df_train["record_id"]]

    df_src_train_labels, df_cat_train_labels = analyze_train_labels(df_train)

    # 6. Render and write markdown report
    md_content = render_eda_markdown(
        df_splits_sources=df_splits_sources,
        df_by_category=df_by_category,
        df_by_source=df_by_source,
        text_stats=text_stats,
        human_stats=human_stats,
        df_llm_variants=df_llm_variants,
        df_disagreement=df_disagreement,
        df_src_train_labels=df_src_train_labels,
        df_cat_train_labels=df_cat_train_labels,
    )

    report_path = args.reports_dir / "eda_report.md"
    report_path.write_text(md_content, encoding="utf-8")

    # 7. Write summary tables to reports/
    df_by_source.to_csv(args.reports_dir / "eda_summary_by_source.csv", index=False)
    df_by_category.to_csv(args.reports_dir / "eda_summary_by_category.csv", index=False)
    df_disagreement.to_csv(args.reports_dir / "eda_disagreement.csv", index=False)
    df_src_train_labels.to_csv(args.reports_dir / "eda_label_distribution.csv", index=False)

    print(f"Wrote EDA report to {report_path}")
    print(f"Wrote summary CSVs to {args.reports_dir}")


if __name__ == "__main__":
    main()
