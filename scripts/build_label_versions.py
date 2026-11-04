"""Build full train set after correction, nested subset ablation versions, and manifests.

Usage:
    python scripts/build_label_versions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.correction.correction import (
    DEFAULT_CORRECTION_LEVELS,
    DEFAULT_SEEDS,
    build_full_train_after_correction,
    build_nested_subset_corrections,
    ensure_train_gold_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label versions and correction manifests.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/dataset.csv")
    parser.add_argument("--llm-train", type=Path, default=ROOT / "data/annotations/llm/llm_train.csv")
    parser.add_argument("--llm-subset", type=Path, default=ROOT / "data/annotations/llm/llm_train_subset.csv")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data/annotations/human/gold")
    parser.add_argument("--out-full-train", type=Path, default=ROOT / "data/processed/train_full_corrected.csv")
    parser.add_argument("--out-manifest", type=Path, default=ROOT / "data/manifests/correction_manifest.csv")
    parser.add_argument("--out-subset-versions", type=Path, default=ROOT / "data/manifests/subset_label_versions.csv")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/correction_report.md")
    return parser.parse_args()


def render_report(df_train_full: pd.DataFrame, manifest_df: pd.DataFrame, stats: dict) -> str:
    lines = [
        "# Label Versions and Correction Manifest Report",
        "",
        "## 1. Full Train Set After Correction Summary",
        "",
        f"- Total records: {len(df_train_full):,}",
        f"- In subset (human gold): {int(df_train_full['is_human_subset'].sum()):,}",
        f"- Outside subset (LLM labels): {int((~df_train_full['is_human_subset']).sum()):,}",
        f"- Final clickbait labels (class 1): {int((df_train_full['label'] == 1).sum()):,} "
        f"({(df_train_full['label'] == 1).mean():.2%})",
        f"- Final non-clickbait labels (class 0): {int((df_train_full['label'] == 0).sum()):,} "
        f"({(df_train_full['label'] == 0).mean():.2%})",
        f"- Total labels corrected from LLM on subset: {int(df_train_full['is_corrected'].sum()):,} "
        f"({df_train_full['is_corrected'].mean():.2%})",
        "",
        "## 2. Nested Correction Ablation Levels on Training Subset",
        "",
        "| Seed | Correction Level | Target Ratio | Corrected Disagreements | Total Subset Records |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for seed, s_info in stats.items():
        for lvl, l_info in s_info["levels"].items():
            lines.append(
                f"| {seed} | {lvl:.0%} | {l_info['target_ratio']:.2f} | "
                f"{l_info['n_corrected']:,} | {l_info['total_records']:,} |"
            )

    lines.extend([
        "",
        "## 3. Outputs and Deliverables",
        "",
        "- `data/processed/train_full_corrected.csv`: 20,000 train records with final corrected labels.",
        "- `data/manifests/correction_manifest.csv`: Audit trail per record, seed, and correction level.",
        "- `data/manifests/subset_label_versions.csv`: Label lookup per record, seed, and correction level.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # 1. Ensure human gold reference for train subset
    df_train_gold = ensure_train_gold_reference(
        dataset_path=args.dataset,
        llm_train_path=args.llm_train,
        gold_dir=args.gold_dir,
    )

    # 2. Build full train after correction (20,000 records)
    df_dataset = pd.read_csv(args.dataset)
    df_llm_train = pd.read_csv(args.llm_train, low_memory=False)
    df_train_full = build_full_train_after_correction(
        df_dataset=df_dataset,
        df_llm_train=df_llm_train,
        df_train_gold=df_train_gold,
    )
    args.out_full_train.parent.mkdir(parents=True, exist_ok=True)
    df_train_full.to_csv(args.out_full_train, index=False)

    # 3. Build nested subset corrections across seeds
    df_subset_records = df_dataset[
        (df_dataset["split"] == "train") & (df_dataset["is_human_subset"] == True)
    ].copy()
    df_llm_subset = pd.read_csv(args.llm_subset)

    manifest_df, versions_df, stats = build_nested_subset_corrections(
        df_subset_records=df_subset_records,
        df_llm_subset=df_llm_subset,
        df_train_gold=df_train_gold,
        levels=DEFAULT_CORRECTION_LEVELS,
        seeds=DEFAULT_SEEDS,
    )

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.out_manifest, index=False)

    args.out_subset_versions.parent.mkdir(parents=True, exist_ok=True)
    versions_df.to_csv(args.out_subset_versions, index=False)

    # 4. Write report
    report_text = render_report(df_train_full, manifest_df, stats)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report_text, encoding="utf-8")

    print(f"Wrote full corrected train to {args.out_full_train}")
    print(f"Wrote correction manifest to {args.out_manifest}")
    print(f"Wrote subset label versions to {args.out_subset_versions}")
    print(f"Wrote correction report to {args.out_report}")


if __name__ == "__main__":
    main()
