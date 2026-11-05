"""Ablation scaling experiments: train models across label-correction levels (0% to 100%).

Evaluates downstream classification on Dev Gold, Test Seen Gold, and Test Unseen Gold
using fixed optimal hyperparameters and decision thresholds selected from Dev tuning (§7.2, §8.1).
Saves fitted models, test predictions, manifests, and plots scaling curves across splits.

Usage:
    python scripts/run_scaling_experiments.py --models lr,lsvm,rf,xgb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.correction.loader import (
    load_dev,
    load_subset,
    load_test,
    load_train_set,
)
from ds_project.features.pipeline import ClickbaitFeaturePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label correction scaling ablation.")
    parser.add_argument(
        "--models",
        type=str,
        default="lr,lsvm,rf,xgb",
        help="Comma-separated models: lr, lsvm, rf, xgb",
    )
    parser.add_argument("--variant", type=str, default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument(
        "--seeds",
        type=str,
        default="2026,42,123",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated correction levels",
    )
    parser.add_argument(
        "--hyperparams-json",
        type=Path,
        default=ROOT / "reports/ml/best_hyperparameters.json",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models/saved_models",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=ROOT / "predictions",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data/manifests/model_manifest.csv",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=ROOT / "reports/ml/scaling_ablation_runs.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "reports/ml/scaling_ablation_summary.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=ROOT / "reports/figures/ml/scaling_ablation_macro_f1.png",
    )
    return parser.parse_args()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "clickbait_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "clickbait_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "clickbait_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def instantiate_model(model_key: str, params: dict[str, Any], seed: int) -> Any:
    if model_key == "LogisticRegression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", None),
            solver=params.get("solver", "lbfgs"),
            max_iter=1000,
            random_state=seed,
        )
    elif model_key == "LinearSVM":
        return SGDClassifier(
            alpha=params.get("alpha", 0.0001),
            loss=params.get("loss", "hinge"),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=2000,
            random_state=seed,
        )
    elif model_key == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 70),
            min_samples_split=params.get("min_samples_split", 5),
            class_weight=params.get("class_weight", None),
            random_state=seed,
            n_jobs=-1,
        )
    elif model_key == "XGBoost":
        return XGBClassifier(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.15),
            n_estimators=params.get("n_estimators", 150),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=params.get("scale_pos_weight", 1.0),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def predict_with_threshold(clf: Any, X: Any, threshold: float, is_svm: bool) -> tuple[np.ndarray, np.ndarray]:
    if is_svm:
        scores = clf.decision_function(X)
        preds = (scores >= threshold).astype(int)
        return preds, scores
    else:
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs


def plot_scaling_curves(
    df_summary: pd.DataFrame,
    full_train_refs: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300, sharey=True)

    model_colors = {
        "LogisticRegression": "#1f77b4",
        "LinearSVM": "#2ca02c",
        "RandomForest": "#ff7f0e",
        "XGBoost": "#d62728",
    }
    model_markers = {
        "LogisticRegression": "o",
        "LinearSVM": "s",
        "RandomForest": "^",
        "XGBoost": "D",
    }

    splits_info = [
        ("dev", "Dev Set (Gold 1,395)", axes[0], "macro_f1_mean", "macro_f1_std"),
        ("test_seen", "Test Seen Set (Gold 998)", axes[1], "test_seen_macro_f1_mean", "test_seen_macro_f1_std"),
        ("test_unseen", "Test Unseen Set (Gold 1,194)", axes[2], "test_unseen_macro_f1_mean", "test_unseen_macro_f1_std"),
    ]

    for split_key, title, ax, mean_col, std_col in splits_info:
        for model_name, grp in df_summary.groupby("model"):
            color = model_colors.get(model_name, "#333333")
            marker = model_markers.get(model_name, "o")
            grp = grp.sort_values("correction_level")

            x = grp["correction_level"] * 100.0
            y = grp[mean_col]
            y_std = grp[std_col]

            ax.plot(
                x,
                y,
                label=f"{model_name}",
                color=color,
                marker=marker,
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=0.15,
            )

            # Horizontal line for full train reference
            if model_name in full_train_refs and split_key in full_train_refs[model_name]:
                ref_val = full_train_refs[model_name][split_key]
                ax.axhline(
                    y=ref_val,
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                    label=f"{model_name} (Full 20k: {ref_val:.4f})",
                )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Correction Level (% Disagreements Corrected)", fontsize=10, labelpad=6)
        if ax == axes[0]:
            ax.set_ylabel("Macro-F1 Score", fontsize=11, labelpad=8)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].legend(loc="lower left", frameon=True, fontsize=8)
    fig.suptitle("Label Correction Scaling vs Downstream Macro-F1 across Splits (EVAL2 Benchmark)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comprehensive scaling plot to {output_path}")


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.runs_csv.parent.mkdir(parents=True, exist_ok=True)

    model_map = {
        "lr": "LogisticRegression",
        "lsvm": "LinearSVM",
        "rf": "RandomForest",
        "xgb": "XGBoost",
    }
    selected_models = [model_map[m.strip().lower()] for m in args.models.split(",") if m.strip().lower() in model_map]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    levels = [float(lvl.strip()) for lvl in args.levels.split(",")]

    print("Loading hyperparameters from:", args.hyperparams_json)
    with args.hyperparams_json.open("r", encoding="utf-8") as f:
        best_hyperparams = json.load(f)

    # 1. Load evaluation datasets
    print("Loading evaluation and test splits...")
    df_dev = load_dev(include_gold=True)
    y_dev = df_dev["label"].to_numpy(dtype=int)

    # Load raw blind test records (for 100% submission prediction outputs)
    df_test_seen_raw = load_test("seen", include_gold=False)
    df_test_unseen_raw = load_test("unseen", include_gold=False)

    # Load test gold labels for ground-truth EVAL2 evaluation
    gold_seen_file = ROOT / "data/annotations/human/gold/test_gold_seen.csv"
    gold_unseen_file = ROOT / "data/annotations/human/gold/test_gold_unseen.csv"

    gold_seen_map = {}
    if gold_seen_file.exists():
        df_g_seen = pd.read_csv(gold_seen_file)
        gold_seen_map = df_g_seen.dropna(subset=["record_id", "final_label"]).set_index("record_id")["final_label"].astype(int).to_dict()

    gold_unseen_map = {}
    if gold_unseen_file.exists():
        df_g_unseen = pd.read_csv(gold_unseen_file)
        gold_unseen_map = df_g_unseen.dropna(subset=["record_id", "final_label"]).set_index("record_id")["final_label"].astype(int).to_dict()

    # Pre-calculate gold alignment masks for evaluation
    seen_eval_indices = [i for i, rid in enumerate(df_test_seen_raw["record_id"]) if rid in gold_seen_map]
    seen_eval_y = np.array([gold_seen_map[df_test_seen_raw.iloc[i]["record_id"]] for i in seen_eval_indices])

    unseen_eval_indices = [i for i, rid in enumerate(df_test_unseen_raw["record_id"]) if rid in gold_unseen_map]
    unseen_eval_y = np.array([gold_unseen_map[df_test_unseen_raw.iloc[i]["record_id"]] for i in unseen_eval_indices])

    # 2. Extract feature matrices on Subset
    base_sub = load_subset(correction_level=1.0, seed=seeds[0])
    print(f"Fitting shared feature pipeline (variant='{args.variant}') on subset ({len(base_sub)} texts)...")
    t0 = time.time()
    pipeline_sub = ClickbaitFeaturePipeline(variant=args.variant)
    X_train_sub = pipeline_sub.fit_transform(base_sub)
    X_dev_sub = pipeline_sub.transform(df_dev)
    X_seen_sub = pipeline_sub.transform(df_test_seen_raw)
    X_unseen_sub = pipeline_sub.transform(df_test_unseen_raw)
    print(f"Subset feature matrices extracted ({time.time() - t0:.1f}s): X_sub={X_train_sub.shape}")

    # 3. Extract feature matrices on Full Train
    print("Fitting shared feature pipeline on Full Train (20,000 texts)...")
    df_full = load_train_set(variant="full_corrected")
    y_full = df_full["label"].to_numpy(dtype=int)
    pipeline_full = ClickbaitFeaturePipeline(variant=args.variant)
    X_train_full = pipeline_full.fit_transform(df_full)
    X_dev_full = pipeline_full.transform(df_dev)
    X_seen_full = pipeline_full.transform(df_test_seen_raw)
    X_unseen_full = pipeline_full.transform(df_test_unseen_raw)

    run_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    full_train_refs: dict[str, dict[str, float]] = {}

    for model_name in selected_models:
        model_cfg = best_hyperparams.get(model_name, {})
        params = model_cfg.get("params", {})
        tau_star = model_cfg.get("optimal_threshold", 0.5)
        is_svm = model_name == "LinearSVM"

        print(f"\n=======================================================")
        print(f"Running Scaling Experiments for: {model_name}")
        print(f"Fixed Params: {params}, Optimal Threshold: {tau_star}")
        print(f"=======================================================")

        # A. Full Train Reference Run
        print(f"Training Full Train Reference (20k records)...")
        clf_full = instantiate_model(model_name, params, seed=2026)
        clf_full.fit(X_train_full, y_full)

        dev_preds_full, _ = predict_with_threshold(clf_full, X_dev_full, tau_star, is_svm)
        m_dev_full = evaluate_predictions(y_dev, dev_preds_full)

        seen_preds_full, seen_scores_full = predict_with_threshold(clf_full, X_seen_full, tau_star, is_svm)
        unseen_preds_full, unseen_scores_full = predict_with_threshold(clf_full, X_unseen_full, tau_star, is_svm)

        m_seen_full = evaluate_predictions(seen_eval_y, seen_preds_full[seen_eval_indices])
        m_unseen_full = evaluate_predictions(unseen_eval_y, unseen_preds_full[unseen_eval_indices])

        full_train_refs[model_name] = {
            "dev": m_dev_full["macro_f1"],
            "test_seen": m_seen_full["macro_f1"],
            "test_unseen": m_unseen_full["macro_f1"],
        }
        print(f"  Full Train Ref -> Dev: {m_dev_full['macro_f1']:.4f} | Seen: {m_seen_full['macro_f1']:.4f} | Unseen: {m_unseen_full['macro_f1']:.4f}")

        # Save full train model and test predictions
        full_model_id = f"{model_name.lower()}_full_train"
        full_model_path = args.models_dir / f"{full_model_id}.joblib"
        joblib.dump({"model": clf_full, "pipeline": pipeline_full, "threshold": tau_star, "params": params}, full_model_path)

        seen_file = args.preds_dir / f"{full_model_id}_test_seen.csv"
        unseen_file = args.preds_dir / f"{full_model_id}_test_unseen.csv"

        pd.DataFrame({"record_id": df_test_seen_raw["record_id"], "predicted_label": seen_preds_full, "confidence_score": seen_scores_full}).to_csv(seen_file, index=False)
        pd.DataFrame({"record_id": df_test_unseen_raw["record_id"], "predicted_label": unseen_preds_full, "confidence_score": unseen_scores_full}).to_csv(unseen_file, index=False)

        manifest_records.append({
            "model_id": full_model_id,
            "model_family": model_name,
            "training_cohort": "full_train_20k",
            "correction_level": 1.0,
            "seed": 2026,
            "features_dim": X_train_full.shape[1],
            "optimal_threshold": tau_star,
            "dev_macro_f1": round(m_dev_full["macro_f1"], 4),
            "dev_accuracy": round(m_dev_full["accuracy"], 4),
            "dev_clickbait_f1": round(m_dev_full["clickbait_f1"], 4),
            "test_seen_macro_f1": round(m_seen_full["macro_f1"], 4),
            "test_seen_accuracy": round(m_seen_full["accuracy"], 4),
            "test_unseen_macro_f1": round(m_unseen_full["macro_f1"], 4),
            "test_unseen_accuracy": round(m_unseen_full["accuracy"], 4),
            "model_path": str(full_model_path.relative_to(ROOT)),
            "test_seen_preds": str(seen_file.relative_to(ROOT)),
            "test_unseen_preds": str(unseen_file.relative_to(ROOT)),
        })

        # B. Scaling Runs across seeds and levels
        for seed in seeds:
            for level in levels:
                t_run = time.time()
                df_sub_slice = load_subset(correction_level=level, seed=seed)
                y_train_sub = df_sub_slice["label"].to_numpy(dtype=int)

                clf = instantiate_model(model_name, params, seed=seed)
                clf.fit(X_train_sub, y_train_sub)
                fit_sec = time.time() - t_run

                dev_preds, _ = predict_with_threshold(clf, X_dev_sub, tau_star, is_svm)
                metrics_dev = evaluate_predictions(y_dev, dev_preds)

                s_preds, s_scores = predict_with_threshold(clf, X_seen_sub, tau_star, is_svm)
                u_preds, u_scores = predict_with_threshold(clf, X_unseen_sub, tau_star, is_svm)

                metrics_seen = evaluate_predictions(seen_eval_y, s_preds[seen_eval_indices])
                metrics_unseen = evaluate_predictions(unseen_eval_y, u_preds[unseen_eval_indices])

                model_id = f"{model_name.lower()}_lvl{int(level*100)}_s{seed}"
                model_file = args.models_dir / f"{model_id}.joblib"
                joblib.dump({"model": clf, "pipeline": pipeline_sub, "threshold": tau_star, "params": params}, model_file)

                s_file = args.preds_dir / f"{model_id}_test_seen.csv"
                u_file = args.preds_dir / f"{model_id}_test_unseen.csv"

                pd.DataFrame({"record_id": df_test_seen_raw["record_id"], "predicted_label": s_preds, "confidence_score": s_scores}).to_csv(s_file, index=False)
                pd.DataFrame({"record_id": df_test_unseen_raw["record_id"], "predicted_label": u_preds, "confidence_score": u_scores}).to_csv(u_file, index=False)

                manifest_records.append({
                    "model_id": model_id,
                    "model_family": model_name,
                    "training_cohort": "subset_6400",
                    "correction_level": level,
                    "seed": seed,
                    "features_dim": X_train_sub.shape[1],
                    "optimal_threshold": tau_star,
                    "dev_macro_f1": round(metrics_dev["macro_f1"], 4),
                    "dev_accuracy": round(metrics_dev["accuracy"], 4),
                    "dev_clickbait_f1": round(metrics_dev["clickbait_f1"], 4),
                    "test_seen_macro_f1": round(metrics_seen["macro_f1"], 4),
                    "test_seen_accuracy": round(metrics_seen["accuracy"], 4),
                    "test_unseen_macro_f1": round(metrics_unseen["macro_f1"], 4),
                    "test_unseen_accuracy": round(metrics_unseen["accuracy"], 4),
                    "model_path": str(model_file.relative_to(ROOT)),
                    "test_seen_preds": str(s_file.relative_to(ROOT)),
                    "test_unseen_preds": str(u_file.relative_to(ROOT)),
                })

                run_records.append({
                    "model": model_name,
                    "correction_level": level,
                    "seed": seed,
                    "fit_time_sec": round(fit_sec, 2),
                    "macro_f1": metrics_dev["macro_f1"],
                    "accuracy": metrics_dev["accuracy"],
                    "clickbait_f1": metrics_dev["clickbait_f1"],
                    "test_seen_macro_f1": metrics_seen["macro_f1"],
                    "test_seen_accuracy": metrics_seen["accuracy"],
                    "test_seen_clickbait_f1": metrics_seen["clickbait_f1"],
                    "test_unseen_macro_f1": metrics_unseen["macro_f1"],
                    "test_unseen_accuracy": metrics_unseen["accuracy"],
                    "test_unseen_clickbait_f1": metrics_unseen["clickbait_f1"],
                })

                print(f"  Level: {int(level*100):3d}% | Seed: {seed:4d} | Dev F1: {metrics_dev['macro_f1']:.4f} | Seen F1: {metrics_seen['macro_f1']:.4f} | Unseen F1: {metrics_unseen['macro_f1']:.4f} | Fit: {fit_sec:.1f}s")

    # Save detailed runs
    df_runs = pd.DataFrame(run_records)
    df_runs.to_csv(args.runs_csv, index=False)
    print(f"\nSaved detailed runs to {args.runs_csv}")

    # Save model manifest
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(args.manifest_path, index=False)
    print(f"Saved model manifest ({len(df_manifest)} entries) to {args.manifest_path}")

    # Compute summary across seeds
    summary_list = []
    for (model, level), grp in df_runs.groupby(["model", "correction_level"]):
        summary_list.append({
            "model": model,
            "correction_level": level,
            "n_seeds": len(grp),
            # Dev metrics
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(),
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(),
            "clickbait_f1_mean": grp["clickbait_f1"].mean(),
            "clickbait_f1_std": grp["clickbait_f1"].std(),
            # Test Seen metrics
            "test_seen_macro_f1_mean": grp["test_seen_macro_f1"].mean(),
            "test_seen_macro_f1_std": grp["test_seen_macro_f1"].std(),
            "test_seen_accuracy_mean": grp["test_seen_accuracy"].mean(),
            "test_seen_accuracy_std": grp["test_seen_accuracy"].std(),
            "test_seen_clickbait_f1_mean": grp["test_seen_clickbait_f1"].mean(),
            "test_seen_clickbait_f1_std": grp["test_seen_clickbait_f1"].std(),
            # Test Unseen metrics
            "test_unseen_macro_f1_mean": grp["test_unseen_macro_f1"].mean(),
            "test_unseen_macro_f1_std": grp["test_unseen_macro_f1"].std(),
            "test_unseen_accuracy_mean": grp["test_unseen_accuracy"].mean(),
            "test_unseen_accuracy_std": grp["test_unseen_accuracy"].std(),
            "test_unseen_clickbait_f1_mean": grp["test_unseen_clickbait_f1"].mean(),
            "test_unseen_clickbait_f1_std": grp["test_unseen_clickbait_f1"].std(),
        })

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(args.summary_csv, index=False)
    print(f"Saved summary to {args.summary_csv}")

    # Plot comprehensive multi-split scaling curves
    plot_scaling_curves(df_summary, full_train_refs, args.plot_path)

    # Print summary table
    print("Ablation Scaling Summary (Mean ± Std over 3 seeds across Dev, Seen, Unseen):")
    for model_name, grp in df_summary.groupby("model"):
        refs = full_train_refs.get(model_name, {})
        print(f"\n--- {model_name} (Full Train 20k -> Dev: {refs.get('dev', 0.0):.4f} | Seen: {refs.get('test_seen', 0.0):.4f} | Unseen: {refs.get('test_unseen', 0.0):.4f}) ---")
        for _, r in grp.sort_values("correction_level").iterrows():
            lvl_pct = int(r["correction_level"] * 100)
            print(f"Lvl {lvl_pct:3d}% | Dev F1: {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | Seen F1: {r['test_seen_macro_f1_mean']:.4f} ± {r['test_seen_macro_f1_std']:.4f} | Unseen F1: {r['test_unseen_macro_f1_mean']:.4f} ± {r['test_unseen_macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
