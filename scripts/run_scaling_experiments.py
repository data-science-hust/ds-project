"""Ablation scaling experiments: train models across label-correction levels (0% to 100%).

Evaluates downstream classification on Dev Gold using fixed optimal hyperparameters
and decision thresholds selected from Dev tuning (§7.2, §8.1).
Saves fitted models, test predictions, manifests, and plots scaling curves.

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
        default=ROOT / "reports/best_hyperparameters.json",
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
        default=ROOT / "reports/scaling_ablation_runs.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "reports/scaling_ablation_summary.csv",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=ROOT / "reports/figures/scaling_ablation_macro_f1.png",
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
            C=params.get("C", 0.5),
            class_weight=params.get("class_weight", None),
            solver=params.get("solver", "lbfgs"),
            max_iter=1000,
            random_state=seed,
        )
    elif model_key == "LinearSVM":
        return SGDClassifier(
            alpha=params.get("alpha", 0.0005),
            loss=params.get("loss", "hinge"),
            class_weight=params.get("class_weight", None),
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
    full_train_refs: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)

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

    for model_name, grp in df_summary.groupby("model"):
        color = model_colors.get(model_name, "#333333")
        marker = model_markers.get(model_name, "o")
        grp = grp.sort_values("correction_level")

        x = grp["correction_level"] * 100.0
        y = grp["macro_f1_mean"]
        y_std = grp["macro_f1_std"]

        plt.plot(
            x,
            y,
            label=f"{model_name} (Subset 6.4k)",
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )
        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=0.15,
        )

        # Plot full train upper reference if available
        if model_name in full_train_refs:
            ref_val = full_train_refs[model_name]
            plt.axhline(
                y=ref_val,
                color=color,
                linestyle="--",
                alpha=0.6,
                label=f"{model_name} (Full Train 20k: {ref_val:.4f})",
            )

    plt.title("Label Correction Scaling vs Downstream Macro-F1 (Evaluated on Dev Gold)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Label Correction Level on 6.4k Subset (% Disagreements Corrected)", fontsize=11, labelpad=8)
    plt.ylabel("Macro-F1 Score", fontsize=11, labelpad=8)
    plt.xticks([0, 25, 50, 75, 100], ["0% (Raw LLM)", "25%", "50%", "75%", "100% (Human Gold)"])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved scaling plot to {output_path}")


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

    # Load baseline datasets
    print("Loading evaluation and test splits...")
    df_dev = load_dev(include_gold=True)
    y_dev = df_dev["label"].to_numpy(dtype=int)

    df_test_seen = load_test("seen")
    df_test_unseen = load_test("unseen")

    # Load base subset text (record_ids and texts are invariant to seed and level)
    base_sub = load_subset(correction_level=1.0, seed=seeds[0])
    print(f"Fitting shared feature pipeline (variant='{args.variant}') on subset ({len(base_sub)} texts)...")
    t0 = time.time()
    pipeline_sub = ClickbaitFeaturePipeline(variant=args.variant)
    X_train_sub = pipeline_sub.fit_transform(base_sub)
    X_dev_sub = pipeline_sub.transform(df_dev)
    X_seen_sub = pipeline_sub.transform(df_test_seen)
    X_unseen_sub = pipeline_sub.transform(df_test_unseen)
    print(f"Subset feature matrices extracted ({time.time() - t0:.1f}s): X_sub={X_train_sub.shape}")

    # Also fit pipeline on Full Train for reference
    print("Fitting shared feature pipeline on Full Train (20,000 texts)...")
    df_full = load_train_set(variant="full_corrected")
    y_full = df_full["label"].to_numpy(dtype=int)
    pipeline_full = ClickbaitFeaturePipeline(variant=args.variant)
    X_train_full = pipeline_full.fit_transform(df_full)
    X_dev_full = pipeline_full.transform(df_dev)
    X_seen_full = pipeline_full.transform(df_test_seen)
    X_unseen_full = pipeline_full.transform(df_test_unseen)

    run_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    full_train_refs: dict[str, float] = {}

    for model_name in selected_models:
        model_cfg = best_hyperparams.get(model_name, {})
        params = model_cfg.get("params", {})
        tau_star = model_cfg.get("optimal_threshold", 0.5)
        is_svm = model_name == "LinearSVM"

        print(f"\n=======================================================")
        print(f"Running Scaling Experiments for: {model_name}")
        print(f"Fixed Params: {params}, Optimal Threshold: {tau_star}")
        print(f"=======================================================")

        # 1. Full Train Reference Run
        print(f"Training Full Train Reference (20k records)...")
        clf_full = instantiate_model(model_name, params, seed=2026)
        clf_full.fit(X_train_full, y_full)

        dev_preds_full, _ = predict_with_threshold(clf_full, X_dev_full, tau_star, is_svm)
        m_full = evaluate_predictions(y_dev, dev_preds_full)
        full_train_refs[model_name] = m_full["macro_f1"]
        print(f"  Full Train Ref: Macro-F1 = {m_full['macro_f1']:.4f}, Accuracy = {m_full['accuracy']:.4f}")

        # Save full train model and test predictions
        full_model_id = f"{model_name.lower()}_full_train"
        full_model_path = args.models_dir / f"{full_model_id}.joblib"
        joblib.dump({"model": clf_full, "pipeline": pipeline_full, "threshold": tau_star, "params": params}, full_model_path)

        seen_preds, seen_scores = predict_with_threshold(clf_full, X_seen_full, tau_star, is_svm)
        unseen_preds, unseen_scores = predict_with_threshold(clf_full, X_unseen_full, tau_star, is_svm)

        seen_file = args.preds_dir / f"{full_model_id}_test_seen.csv"
        unseen_file = args.preds_dir / f"{full_model_id}_test_unseen.csv"

        df_seen_out = pd.DataFrame({"record_id": df_test_seen["record_id"], "predicted_label": seen_preds, "confidence_score": seen_scores})
        df_seen_out.to_csv(seen_file, index=False)

        df_unseen_out = pd.DataFrame({"record_id": df_test_unseen["record_id"], "predicted_label": unseen_preds, "confidence_score": unseen_scores})
        df_unseen_out.to_csv(unseen_file, index=False)

        manifest_records.append({
            "model_id": full_model_id,
            "model_family": model_name,
            "training_cohort": "full_train_20k",
            "correction_level": 1.0,
            "seed": 2026,
            "features_dim": X_train_full.shape[1],
            "optimal_threshold": tau_star,
            "dev_macro_f1": round(m_full["macro_f1"], 4),
            "dev_accuracy": round(m_full["accuracy"], 4),
            "dev_clickbait_f1": round(m_full["clickbait_f1"], 4),
            "model_path": str(full_model_path.relative_to(ROOT)),
            "test_seen_preds": str(seen_file.relative_to(ROOT)),
            "test_unseen_preds": str(unseen_file.relative_to(ROOT)),
        })

        # 2. Scaling Runs across seeds and levels
        for seed in seeds:
            for level in levels:
                t_run = time.time()
                df_sub_slice = load_subset(correction_level=level, seed=seed)
                y_train_sub = df_sub_slice["label"].to_numpy(dtype=int)

                clf = instantiate_model(model_name, params, seed=seed)
                clf.fit(X_train_sub, y_train_sub)
                fit_sec = time.time() - t_run

                dev_preds, _ = predict_with_threshold(clf, X_dev_sub, tau_star, is_svm)
                metrics = evaluate_predictions(y_dev, dev_preds)

                # Predictions on Test Seen and Unseen
                s_preds, s_scores = predict_with_threshold(clf, X_seen_sub, tau_star, is_svm)
                u_preds, u_scores = predict_with_threshold(clf, X_unseen_sub, tau_star, is_svm)

                model_id = f"{model_name.lower()}_lvl{int(level*100)}_s{seed}"
                model_file = args.models_dir / f"{model_id}.joblib"
                joblib.dump({"model": clf, "pipeline": pipeline_sub, "threshold": tau_star, "params": params}, model_file)

                s_file = args.preds_dir / f"{model_id}_test_seen.csv"
                u_file = args.preds_dir / f"{model_id}_test_unseen.csv"

                pd.DataFrame({"record_id": df_test_seen["record_id"], "predicted_label": s_preds, "confidence_score": s_scores}).to_csv(s_file, index=False)
                pd.DataFrame({"record_id": df_test_unseen["record_id"], "predicted_label": u_preds, "confidence_score": u_scores}).to_csv(u_file, index=False)

                manifest_records.append({
                    "model_id": model_id,
                    "model_family": model_name,
                    "training_cohort": "subset_6400",
                    "correction_level": level,
                    "seed": seed,
                    "features_dim": X_train_sub.shape[1],
                    "optimal_threshold": tau_star,
                    "dev_macro_f1": round(metrics["macro_f1"], 4),
                    "dev_accuracy": round(metrics["accuracy"], 4),
                    "dev_clickbait_f1": round(metrics["clickbait_f1"], 4),
                    "model_path": str(model_file.relative_to(ROOT)),
                    "test_seen_preds": str(s_file.relative_to(ROOT)),
                    "test_unseen_preds": str(u_file.relative_to(ROOT)),
                })

                run_records.append({
                    "model": model_name,
                    "correction_level": level,
                    "seed": seed,
                    "fit_time_sec": round(fit_sec, 2),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "clickbait_f1": metrics["clickbait_f1"],
                    "clickbait_precision": metrics["clickbait_precision"],
                    "clickbait_recall": metrics["clickbait_recall"],
                })

                print(f"  Level: {int(level*100):3d}% | Seed: {seed:4d} | Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} | Fit: {fit_sec:.1f}s")

    # Save detailed runs
    df_runs = pd.DataFrame(run_records)
    df_runs.to_csv(args.runs_csv, index=False)
    print(f"\nSaved runs to {args.runs_csv}")

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
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(),
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(),
            "clickbait_f1_mean": grp["clickbait_f1"].mean(),
            "clickbait_f1_std": grp["clickbait_f1"].std(),
            "clickbait_recall_mean": grp["clickbait_recall"].mean(),
            "clickbait_recall_std": grp["clickbait_recall"].std(),
            "clickbait_precision_mean": grp["clickbait_precision"].mean(),
            "clickbait_precision_std": grp["clickbait_precision"].std(),
        })

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(args.summary_csv, index=False)
    print(f"Saved summary to {args.summary_csv}")

    # Plot
    plot_scaling_curves(df_summary, full_train_refs, args.plot_path)

    # Print summary table
    print("\n=======================================================")
    print("Ablation Scaling Summary (Mean ± Std over 3 seeds on Dev Gold):")
    print("=======================================================")
    for model_name, grp in df_summary.groupby("model"):
        print(f"\n--- {model_name} (Upper bound Full Train 20k: {full_train_refs.get(model_name, 0.0):.4f}) ---")
        for _, r in grp.sort_values("correction_level").iterrows():
            lvl_pct = int(r["correction_level"] * 100)
            print(f"Level {lvl_pct:3d}%: Macro-F1 = {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | Clickbait-F1 = {r['clickbait_f1_mean']:.4f} ± {r['clickbait_f1_std']:.4f} | Accuracy = {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}")


if __name__ == "__main__":
    main()
