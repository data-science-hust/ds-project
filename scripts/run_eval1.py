import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.evaluation.eval1 import (
    VARIANTS,
    analyze_human,
    calculate_metrics,
    check_gold,
    paired_bootstrap,
    prepare_llm,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EVAL1 on DEV data."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/processed/dataset.csv",
    )

    parser.add_argument(
        "--dev-gold",
        type=Path,
        default=ROOT / "data/annotations/human/gold/dev_gold.csv",
    )

    parser.add_argument(
        "--adjudication-log",
        type=Path,
        default=ROOT / "data/annotations/human/dev/adjudication_log.csv",
    )

    parser.add_argument(
        "--llm-dir",
        type=Path,
        default=ROOT / "data/annotations/llm",
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports/eval1",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # Load DEV and human gold
    df_dataset = pd.read_csv(args.dataset)

    df_dev = df_dataset[df_dataset["split"] == "dev"][
        ["record_id", "source", "category", "title"]
    ].copy()

    df_gold = pd.read_csv(args.dev_gold)

    df_gold, df_missing, gold_summary = check_gold(
        df_dev, df_gold
    )

    pd.DataFrame([gold_summary]).to_csv(
        args.reports_dir / "gold_summary.csv",
        index=False,
    )

    df_missing.to_csv(
        args.reports_dir / "missing_dev_gold.csv",
        index=False,
    )

    print(
        f"DEV: {len(df_dev)} records, "
        f"{len(df_gold)} gold labels, "
        f"{len(df_missing)} missing"
    )

    # Human-human agreement
    if args.adjudication_log.exists():
        df_adj = pd.read_csv(args.adjudication_log)

        df_human, human_summary = analyze_human(df_adj)

        df_human.to_csv(
            args.reports_dir / "human_agreement.csv",
            index=False,
        )

        pd.DataFrame([human_summary]).to_csv(
            args.reports_dir / "human_summary.csv",
            index=False,
        )

        print(
            f"Human agreement: "
            f"{human_summary['raw_agreement']:.4f}"
        )

    else:
        print(
            f"Skipping human agreement: "
            f"{args.adjudication_log} not found"
        )

    # Evaluate four LLM variants
    predictions = {}
    metrics_rows = []
    coverage_rows = []
    group_rows = []

    for variant in VARIANTS:
        llm_path = args.llm_dir / f"llm_dev_{variant}.csv"

        if not llm_path.exists():
            print(f"Skipping {variant}: {llm_path} not found")
            continue

        df_llm = pd.read_csv(llm_path)

        df_eval, coverage = prepare_llm(
            df_llm, df_dev, df_gold, variant
        )

        coverage_rows.append(coverage)
        predictions[variant] = df_eval

        if df_eval.empty:
            print(f"{variant}: no valid predictions with gold")
            continue

        scores = calculate_metrics(
            df_eval["final_label"],
            df_eval["pred"],
        )

        metrics_rows.append({
            "variant": variant,
            **scores,
        })

        print(
            f"{variant}: macro-F1={scores['macro_f1']:.4f}, "
            f"coverage={coverage['coverage']:.2%}"
        )

        # Metrics by source and category
        for dimension in ["source", "category"]:
            for name, group in df_eval.groupby(dimension):
                group_scores = calculate_metrics(
                    group["final_label"],
                    group["pred"],
                )

                group_rows.append({
                    "variant": variant,
                    "dimension": dimension,
                    "group": name,
                    **group_scores,
                })

        # Save incorrect predictions
        df_errors = df_eval[
            df_eval["final_label"] != df_eval["pred"]
        ]

        df_errors.to_csv(
            args.reports_dir / f"errors_{variant}.csv",
            index=False,
        )

    pd.DataFrame(metrics_rows).to_csv(
        args.reports_dir / "llm_metrics.csv",
        index=False,
    )

    pd.DataFrame(coverage_rows).to_csv(
        args.reports_dir / "coverage.csv",
        index=False,
    )

    pd.DataFrame(group_rows).to_csv(
        args.reports_dir / "metrics_by_group.csv",
        index=False,
    )

    # Paired bootstrap
    bootstrap_rows = []

    for i, variant_a in enumerate(VARIANTS):
        for variant_b in VARIANTS[i + 1:]:
            if (
                variant_a not in predictions
                or variant_b not in predictions
            ):
                continue

            if (
                predictions[variant_a].empty
                or predictions[variant_b].empty
            ):
                continue

            result = paired_bootstrap(
                predictions[variant_a],
                predictions[variant_b],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )

            bootstrap_rows.append({
                "config_a": variant_a,
                "config_b": variant_b,
                **result,
            })

    pd.DataFrame(bootstrap_rows).to_csv(
        args.reports_dir / "paired_bootstrap.csv",
        index=False,
    )

    print(f"Wrote EVAL1 results to {args.reports_dir}")


if __name__ == "__main__":
    main()