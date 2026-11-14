"""Pure-CPU evaluation, 3-seed summary, paired bootstrap vs ML, and manifest upsert for XLM-R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ds_project.correction.loader import load_test
from ds_project.evaluation.eval1 import paired_bootstrap
from ds_project.modeling.xlmr import (
    MODEL_MANIFEST_COLUMNS,
    XLMR_HIDDEN_SIZE,
    compute_xlmr_metrics,
    upsert_model_manifest,
)

DEFAULT_ML_BASELINES = (
    "logisticregression_full_train",
    "linearsvm_full_train",
    "randomforest_full_train",
    "xgboost_full_train",
)

SUMMARY_METRICS = (
    "optimal_threshold",
    "dev_macro_f1",
    "dev_accuracy",
    "dev_kappa",
    "dev_macro_precision",
    "dev_macro_recall",
    "dev_precision_0",
    "dev_recall_0",
    "dev_f1_0",
    "dev_precision_1",
    "dev_recall_1",
    "dev_f1_1",
    "test_seen_macro_f1",
    "test_seen_accuracy",
    "test_seen_kappa",
    "test_seen_macro_precision",
    "test_seen_macro_recall",
    "test_seen_precision_0",
    "test_seen_recall_0",
    "test_seen_f1_0",
    "test_seen_precision_1",
    "test_seen_recall_1",
    "test_seen_f1_1",
    "test_unseen_macro_f1",
    "test_unseen_accuracy",
    "test_unseen_kappa",
    "test_unseen_macro_precision",
    "test_unseen_macro_recall",
    "test_unseen_precision_0",
    "test_unseen_recall_0",
    "test_unseen_f1_0",
    "test_unseen_precision_1",
    "test_unseen_recall_1",
    "test_unseen_f1_1",
    "seen_unseen_macro_f1_gap",
    "seen_unseen_accuracy_gap",
)


def _to_rel_posix(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_pred_path(preds_dir: Path, model_id: str, split_key: str, fallback_rel: str | None) -> Path:
    preferred = preds_dir / f"{model_id}_test_{split_key}.csv"
    if preferred.exists():
        return preferred
    if fallback_rel and not pd.isna(fallback_rel):
        cand = Path(str(fallback_rel))
        if not cand.is_absolute():
            cand = PROJECT_ROOT / cand
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Prediction CSV not found for {model_id} test_{split_key}: tried {preferred}")


def _load_and_validate_preds(
    pred_path: Path,
    unlabeled_df: pd.DataFrame | None,
) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = ["record_id", "predicted_label", "confidence_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction file {pred_path} missing required columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError(f"Prediction file {pred_path} contains NaN values")

    df = df.copy()
    df["record_id"] = df["record_id"].astype(str)
    if df["record_id"].duplicated().any():
        raise ValueError(f"Prediction file {pred_path} contains duplicate record_id rows")

    unique_preds = set(df["predicted_label"].astype(int).unique())
    if not unique_preds.issubset({0, 1}):
        raise ValueError(f"Prediction file {pred_path} contains invalid labels: {unique_preds}")
    df["predicted_label"] = df["predicted_label"].astype(int)

    if unlabeled_df is not None:
        expected_ids = set(unlabeled_df["record_id"].astype(str))
        actual_ids = set(df["record_id"])
        if len(df) != len(unlabeled_df) or actual_ids != expected_ids:
            raise ValueError(
                f"Prediction IDs in {pred_path} (n={len(df)}) do not match "
                f"expected unlabeled split (n={len(unlabeled_df)})"
            )
    return df


def _evaluate_split_against_gold(
    preds_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    split_name: str,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    gold_clean = gold_df[["record_id", "label"]].copy()
    gold_clean["record_id"] = gold_clean["record_id"].astype(str)
    gold_clean["label"] = gold_clean["label"].astype(int)

    merged = gold_clean.merge(preds_df[["record_id", "predicted_label"]], on="record_id", how="inner")
    if len(merged) != len(gold_clean):
        raise ValueError(
            f"Only {len(merged)}/{len(gold_clean)} gold rows matched predictions for {split_name}"
        )

    metrics = compute_xlmr_metrics(merged["label"].to_numpy(), merged["predicted_label"].to_numpy())
    bootstrap_df = merged.rename(columns={"label": "final_label", "predicted_label": "pred"})[
        ["record_id", "final_label", "pred"]
    ]
    return metrics, bootstrap_df


def _prefix_metrics(metrics: dict[str, float | int], prefix: str) -> dict[str, float | int]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _build_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in SUMMARY_METRICS:
        if metric not in runs_df.columns:
            continue
        vals = runs_df[metric].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        rows.append({
            "metric": metric,
            "n_seeds": int(vals.size),
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean_plus_minus_std": f"{mean_val:.4f} ± {std_val:.4f}",
        })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows available._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                cells.append("")
            elif isinstance(val, (float, np.floating)):
                fval = float(val)
                if fval != 0.0 and abs(fval) < 1e-3:
                    cells.append(f"{fval:.2e}")
                else:
                    cells.append(format(fval, float_fmt))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_report(
    *,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    tuning_csv: Path | None,
    preproc_report_path: Path | None,
    manifest_df: pd.DataFrame,
) -> str:
    lines: list[str] = ["# XLM-RoBERTa-large Evaluation Report", ""]

    first_run = runs_df.iloc[0]
    lines.extend([
        "## 1. Model & Training Configuration",
        f"- **Model:** `{first_run.get('model_name', 'xlm-roberta-large')}` (`features_dim={XLMR_HIDDEN_SIZE}`)",
        f"- **Training cohort:** `{first_run.get('training_cohort', 'full_train_20k')}` (`train_full_corrected.csv`, 20,000 rows)",
        f"- **Learning rate:** `{first_run.get('lr')}` | **Class weight:** `{first_run.get('class_weight')}` | **Epochs:** `{first_run.get('epochs')}` | **Effective batch size:** `{first_run.get('effective_batch_size')}`",
        "",
    ])

    if preproc_report_path is not None and preproc_report_path.exists():
        pre = json.loads(preproc_report_path.read_text(encoding="utf-8"))
        lines.append(f"- **Preprocessing `max_length`:** `{pre.get('max_length')}` (selected on Train + Dev only)")
        split_rows = []
        for split_name, st in pre.get("splits", {}).items():
            split_rows.append({
                "split": split_name,
                "count": st.get("count"),
                "min": st.get("min"),
                "median": st.get("median"),
                "p95": st.get("p95"),
                "p99": st.get("p99"),
                "max": st.get("max"),
                "truncation_rate": st.get("truncation_rate"),
            })
        if split_rows:
            lines.extend(["", _df_to_markdown(pd.DataFrame(split_rows)), ""])

    lines.append("## 2. Dev Hyperparameter Tuning")
    if tuning_csv is not None and tuning_csv.exists():
        tdf = pd.read_csv(tuning_csv).sort_values("dev_macro_f1", ascending=False)
        cols = [
            c
            for c in [
                "config_id",
                "lr",
                "class_weight",
                "best_epoch",
                "dev_macro_f1_050",
                "optimal_threshold",
                "dev_macro_f1",
                "dev_accuracy",
                "train_seconds",
            ]
            if c in tdf.columns
        ]
        lines.extend(["", _df_to_markdown(tdf[cols]), ""])
    else:
        lines.extend(["", "_Tuning results file not found or skipped via explicit override._", ""])

    lines.append("## 3. Per-Seed Results (Dev, Test Seen, Test Unseen)")
    run_cols = [
        c
        for c in [
            "model_id",
            "seed",
            "best_epoch",
            "optimal_threshold",
            "dev_macro_f1",
            "dev_accuracy",
            "test_seen_macro_f1",
            "test_seen_accuracy",
            "test_seen_precision_1",
            "test_seen_recall_1",
            "test_seen_f1_1",
            "test_unseen_macro_f1",
            "test_unseen_accuracy",
            "test_unseen_precision_1",
            "test_unseen_recall_1",
            "test_unseen_f1_1",
            "seen_unseen_macro_f1_gap",
        ]
        if c in runs_df.columns
    ]
    lines.extend(["", _df_to_markdown(runs_df[run_cols]), ""])

    lines.append("## 4. Confusion Matrices (Test Splits)")
    cm_cols = [
        c
        for c in [
            "model_id",
            "test_seen_tn",
            "test_seen_fp",
            "test_seen_fn",
            "test_seen_tp",
            "test_unseen_tn",
            "test_unseen_fp",
            "test_unseen_fn",
            "test_unseen_tp",
        ]
        if c in runs_df.columns
    ]
    lines.extend(["", _df_to_markdown(runs_df[cm_cols]), ""])

    lines.append("## 5. Multi-Seed Summary (Mean ± Std)")
    key_summary = summary_df[
        summary_df["metric"].isin([
            "optimal_threshold",
            "dev_macro_f1",
            "dev_accuracy",
            "dev_f1_0",
            "dev_f1_1",
            "test_seen_macro_f1",
            "test_seen_accuracy",
            "test_seen_precision_1",
            "test_seen_recall_1",
            "test_seen_f1_1",
            "test_unseen_macro_f1",
            "test_unseen_accuracy",
            "test_unseen_precision_1",
            "test_unseen_recall_1",
            "test_unseen_f1_1",
            "seen_unseen_macro_f1_gap",
            "seen_unseen_accuracy_gap",
        ])
    ]
    lines.extend(["", _df_to_markdown(key_summary), ""])

    lines.append("## 6. Paired Bootstrap Comparison vs Full-Train ML Baselines (95% CI)")
    lines.extend(["", _df_to_markdown(bootstrap_df), ""])

    lines.append("## 7. Manifest Rows Registered")
    lines.extend(["", _df_to_markdown(manifest_df[["model_id", "seed", "optimal_threshold", "dev_macro_f1", "test_seen_macro_f1", "test_unseen_macro_f1", "model_path"]]), ""])

    return "\n".join(lines)


def run_evaluation(
    *,
    runs_dev_csv: Path,
    preds_dir: Path,
    reports_dir: Path,
    manifest_path: Path,
    tuning_csv: Path | None = None,
    preproc_report_path: Path | None = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 2026,
    baseline_ids: tuple[str, ...] = DEFAULT_ML_BASELINES,
    test_seen_unlabeled: pd.DataFrame | None = None,
    test_seen_gold: pd.DataFrame | None = None,
    test_unseen_unlabeled: pd.DataFrame | None = None,
    test_unseen_gold: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate server XLM-R predictions against local gold splits and upsert model_manifest.csv."""
    if not runs_dev_csv.exists():
        raise FileNotFoundError(f"Dev runs CSV not found: {runs_dev_csv}")
    runs_dev_df = pd.read_csv(runs_dev_csv)
    if runs_dev_df.empty:
        raise ValueError(f"Dev runs CSV is empty: {runs_dev_csv}")

    seen_unlabeled = test_seen_unlabeled if test_seen_unlabeled is not None else load_test("seen", include_gold=False)
    seen_gold = test_seen_gold if test_seen_gold is not None else load_test("seen", include_gold=True)
    unseen_unlabeled = (
        test_unseen_unlabeled if test_unseen_unlabeled is not None else load_test("unseen", include_gold=False)
    )
    unseen_gold = test_unseen_gold if test_unseen_gold is not None else load_test("unseen", include_gold=True)

    evaluated_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    xlmr_bootstrap_frames: dict[str, dict[str, pd.DataFrame]] = {}

    for _, dev_row in runs_dev_df.iterrows():
        model_id = str(dev_row["model_id"])
        seen_pred_path = _resolve_pred_path(preds_dir, model_id, "seen", dev_row.get("test_seen_preds"))
        unseen_pred_path = _resolve_pred_path(preds_dir, model_id, "unseen", dev_row.get("test_unseen_preds"))

        seen_preds_df = _load_and_validate_preds(seen_pred_path, seen_unlabeled)
        unseen_preds_df = _load_and_validate_preds(unseen_pred_path, unseen_unlabeled)

        seen_metrics, seen_boot_df = _evaluate_split_against_gold(seen_preds_df, seen_gold, "test_seen")
        unseen_metrics, unseen_boot_df = _evaluate_split_against_gold(unseen_preds_df, unseen_gold, "test_unseen")
        xlmr_bootstrap_frames[model_id] = {"test_seen": seen_boot_df, "test_unseen": unseen_boot_df}

        combined_row = dev_row.to_dict()
        combined_row.update(_prefix_metrics(seen_metrics, "test_seen"))
        combined_row.update(_prefix_metrics(unseen_metrics, "test_unseen"))
        combined_row["seen_unseen_macro_f1_gap"] = float(seen_metrics["macro_f1"]) - float(unseen_metrics["macro_f1"])
        combined_row["seen_unseen_accuracy_gap"] = float(seen_metrics["accuracy"]) - float(unseen_metrics["accuracy"])
        combined_row["test_seen_preds"] = _to_rel_posix(seen_pred_path)
        combined_row["test_unseen_preds"] = _to_rel_posix(unseen_pred_path)
        evaluated_rows.append(combined_row)

        manifest_rows.append({
            "model_id": model_id,
            "model_family": str(dev_row.get("model_family", "XLMR-large")),
            "training_cohort": str(dev_row.get("training_cohort", "full_train_20k")),
            "correction_level": float(dev_row.get("correction_level", 1.0)),
            "seed": int(dev_row["seed"]),
            "features_dim": int(dev_row.get("features_dim", XLMR_HIDDEN_SIZE)),
            "optimal_threshold": float(dev_row["optimal_threshold"]),
            "dev_macro_f1": float(dev_row["dev_macro_f1"]),
            "dev_accuracy": float(dev_row["dev_accuracy"]),
            "dev_clickbait_f1": float(dev_row["dev_f1_1"]),
            "test_seen_macro_f1": float(seen_metrics["macro_f1"]),
            "test_seen_accuracy": float(seen_metrics["accuracy"]),
            "test_unseen_macro_f1": float(unseen_metrics["macro_f1"]),
            "test_unseen_accuracy": float(unseen_metrics["accuracy"]),
            "model_path": str(dev_row["model_path"]),
            "test_seen_preds": _to_rel_posix(seen_pred_path),
            "test_unseen_preds": _to_rel_posix(unseen_pred_path),
        })

    runs_df = pd.DataFrame(evaluated_rows)
    summary_df = _build_summary(runs_df)

    # Paired bootstrap comparison for primary seed against available full-train ML baselines
    primary_model_id = (
        "xlmrlarge_full_train"
        if "xlmrlarge_full_train" in xlmr_bootstrap_frames
        else str(runs_df.iloc[0]["model_id"])
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for split_key, gold_df in (("seen", seen_gold), ("unseen", unseen_gold)):
        split_name = f"test_{split_key}"
        xlmr_df = xlmr_bootstrap_frames[primary_model_id][split_name]
        for base_id in baseline_ids:
            base_path = preds_dir / f"{base_id}_test_{split_key}.csv"
            if not base_path.exists():
                continue
            base_preds_df = _load_and_validate_preds(base_path, None)
            _, base_boot_df = _evaluate_split_against_gold(base_preds_df, gold_df, split_name)
            boot = paired_bootstrap(base_boot_df, xlmr_df, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
            ci_low = float(boot["ci_low"])
            ci_high = float(boot["ci_high"])
            bootstrap_rows.append({
                "split": split_name,
                "xlmr_model_id": primary_model_id,
                "baseline_model_id": base_id,
                "n_common": int(boot["n_common"]),
                "macro_f1_baseline": float(boot["macro_f1_a"]),
                "macro_f1_xlmr": float(boot["macro_f1_b"]),
                "delta_macro_f1": float(boot["delta_macro_f1"]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant_95": bool(ci_low > 0.0 or ci_high < 0.0),
            })

    boot_cols = [
        "split",
        "xlmr_model_id",
        "baseline_model_id",
        "n_common",
        "macro_f1_baseline",
        "macro_f1_xlmr",
        "delta_macro_f1",
        "ci_low",
        "ci_high",
        "significant_95",
    ]
    bootstrap_df = pd.DataFrame(bootstrap_rows, columns=boot_cols)

    reports_dir.mkdir(parents=True, exist_ok=True)
    runs_path = reports_dir / "xlmr_runs.csv"
    summary_path = reports_dir / "xlmr_summary.csv"
    bootstrap_path = reports_dir / "xlmr_bootstrap_vs_ml.csv"
    xlmr_manifest_path = reports_dir / "xlmr_manifest.csv"
    report_md_path = reports_dir / "xlmr_report.md"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)

    manifest_df = pd.DataFrame(manifest_rows, columns=list(MODEL_MANIFEST_COLUMNS))
    manifest_df.to_csv(xlmr_manifest_path, index=False)
    combined_manifest = upsert_model_manifest(manifest_path, manifest_rows)

    report_md = _render_report(
        runs_df=runs_df,
        summary_df=summary_df,
        bootstrap_df=bootstrap_df,
        tuning_csv=tuning_csv,
        preproc_report_path=preproc_report_path,
        manifest_df=manifest_df,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    return {
        "runs_path": str(runs_path),
        "summary_path": str(summary_path),
        "bootstrap_path": str(bootstrap_path),
        "xlmr_manifest_path": str(xlmr_manifest_path),
        "manifest_path": str(manifest_path),
        "report_md_path": str(report_md_path),
        "n_runs": len(runs_df),
        "manifest_total_rows": len(combined_manifest),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XLM-R predictions on CPU and update manifest.")
    parser.add_argument("--runs-dev-csv", type=Path, default=PROJECT_ROOT / "reports/xlmr/xlmr_runs_dev.csv")
    parser.add_argument("--tuning-csv", type=Path, default=PROJECT_ROOT / "reports/xlmr/xlmr_tuning_results.csv")
    parser.add_argument(
        "--preproc-report",
        type=Path,
        default=PROJECT_ROOT / "reports/xlmr/xlmr_preprocessing_report.json",
    )
    parser.add_argument("--preds-dir", type=Path, default=PROJECT_ROOT / "predictions")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports/xlmr")
    parser.add_argument("--manifest-path", type=Path, default=PROJECT_ROOT / "data/manifests/model_manifest.csv")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = run_evaluation(
        runs_dev_csv=args.runs_dev_csv,
        preds_dir=args.preds_dir,
        reports_dir=args.reports_dir,
        manifest_path=args.manifest_path,
        tuning_csv=args.tuning_csv,
        preproc_report_path=args.preproc_report,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Evaluated {out['n_runs']} XLM-R run(s). Report -> {out['report_md_path']}")
    print(f"Updated manifest ({out['manifest_total_rows']} total rows) -> {out['manifest_path']}")


if __name__ == "__main__":
    main()
