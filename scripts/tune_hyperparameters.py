"""Hyperparameter optimization on the Dev set for top 3 ML models: LR, LSVM, XGBoost.

Evaluates configurations on Dev set to find optimal hyperparameters and decision thresholds
strictly adhering to the project's data contract (§8.1: Dev selection only).

Usage:
    python scripts/tune_hyperparameters.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.correction.loader import load_dev, load_train_set
from ds_project.features.pipeline import ClickbaitFeaturePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hyperparameters for top models.")
    parser.add_argument("--variant", type=str, default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-csv", type=Path, default=ROOT / "reports/ml/hyperparameter_tuning_results.csv")
    parser.add_argument("--out-json", type=Path, default=ROOT / "reports/ml/best_hyperparameters.json")
    return parser.parse_args()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "clickbait_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "clickbait_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "clickbait_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Scan probability / decision score thresholds on Dev to find maximum Macro-F1."""
    best_th = 0.5
    best_metrics: dict[str, float] = {}
    best_f1 = -1.0

    thresholds = np.linspace(0.30, 0.70, 41)
    for th in thresholds:
        preds = (scores >= th).astype(int)
        m = evaluate_predictions(y_true, preds)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_th = float(th)
            best_metrics = m

    return best_th, best_metrics


def main() -> None:
    args = parse_args()

    print(f"Loading datasets for tuning on variant '{args.variant}'...")
    df_train = load_train_set(variant="full_corrected")
    df_dev = load_dev(include_gold=True)

    y_train = df_train["label"].to_numpy(dtype=int)
    y_dev = df_dev["label"].to_numpy(dtype=int)

    print("Extracting feature matrices...")
    t0 = time.time()
    pipeline = ClickbaitFeaturePipeline(variant=args.variant)
    X_train = pipeline.fit_transform(df_train)
    X_dev = pipeline.transform(df_dev)
    print(f"Feature matrix ready: X_train={X_train.shape}, X_dev={X_dev.shape} ({time.time() - t0:.1f}s)")

    records: list[dict[str, Any]] = []
    best_configs: dict[str, Any] = {}

    # 1. LOGISTIC REGRESSION TUNING
    print("\n--- Tuning Logistic Regression ---")
    lr_grid = [
        {"C": 0.1, "class_weight": None, "solver": "lbfgs"},
        {"C": 0.5, "class_weight": None, "solver": "lbfgs"},
        {"C": 1.0, "class_weight": None, "solver": "lbfgs"},
        {"C": 2.0, "class_weight": None, "solver": "lbfgs"},
        {"C": 3.0, "class_weight": None, "solver": "lbfgs"},
        {"C": 5.0, "class_weight": None, "solver": "lbfgs"},
        {"C": 10.0, "class_weight": None, "solver": "lbfgs"},
        {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs"},
        {"C": 2.0, "class_weight": "balanced", "solver": "lbfgs"},
    ]

    best_lr_score = -1.0
    best_lr_item: dict[str, Any] = {}

    for cfg in lr_grid:
        t_start = time.time()
        clf = LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            solver=cfg["solver"],
            max_iter=1000,
            random_state=args.seed,
        )
        clf.fit(X_train, y_train)
        fit_dur = time.time() - t_start

        probs = clf.predict_proba(X_dev)[:, 1]
        default_preds = (probs >= 0.5).astype(int)
        default_m = evaluate_predictions(y_dev, default_preds)

        opt_th, opt_m = tune_threshold(y_dev, probs)

        row = {
            "model": "LogisticRegression",
            "params": str(cfg),
            "fit_time_sec": round(fit_dur, 2),
            "default_macro_f1": default_m["macro_f1"],
            "default_accuracy": default_m["accuracy"],
            "optimal_threshold": round(opt_th, 3),
            "tuned_macro_f1": opt_m["macro_f1"],
            "tuned_accuracy": opt_m["accuracy"],
            "tuned_clickbait_f1": opt_m["clickbait_f1"],
            "tuned_clickbait_recall": opt_m["clickbait_recall"],
            "tuned_clickbait_precision": opt_m["clickbait_precision"],
        }
        records.append(row)

        if opt_m["macro_f1"] > best_lr_score:
            best_lr_score = opt_m["macro_f1"]
            best_lr_item = {
                "params": cfg,
                "optimal_threshold": round(opt_th, 3),
                "metrics": opt_m,
            }

    print(f"Best LR: Macro-F1 = {best_lr_score:.4f} (params={best_lr_item['params']}, th={best_lr_item['optimal_threshold']})")
    best_configs["LogisticRegression"] = best_lr_item

    # 2. LINEAR SVM TUNING
    print("\n--- Tuning Linear SVM ---")
    lsvm_grid = [
        {"alpha": 1e-5, "loss": "hinge", "class_weight": None},
        {"alpha": 2e-5, "loss": "hinge", "class_weight": None},
        {"alpha": 5e-5, "loss": "hinge", "class_weight": None},
        {"alpha": 1e-4, "loss": "hinge", "class_weight": None},
        {"alpha": 2e-4, "loss": "hinge", "class_weight": None},
        {"alpha": 5e-4, "loss": "hinge", "class_weight": None},
        {"alpha": 1e-4, "loss": "hinge", "class_weight": "balanced"},
        {"alpha": 5e-5, "loss": "modified_huber", "class_weight": None},
        {"alpha": 1e-4, "loss": "modified_huber", "class_weight": None},
    ]

    best_lsvm_score = -1.0
    best_lsvm_item: dict[str, Any] = {}

    for cfg in lsvm_grid:
        t_start = time.time()
        clf = SGDClassifier(
            alpha=cfg["alpha"],
            loss=cfg["loss"],
            class_weight=cfg["class_weight"],
            max_iter=2000,
            random_state=args.seed,
        )
        clf.fit(X_train, y_train)
        fit_dur = time.time() - t_start

        if cfg["loss"] == "modified_huber":
            scores = clf.predict_proba(X_dev)[:, 1]
            opt_th, opt_m = tune_threshold(y_dev, scores)
        else:
            dec = clf.decision_function(X_dev)
            # Threshold scan on raw decision margin
            best_th = 0.0
            best_m = evaluate_predictions(y_dev, (dec >= 0.0).astype(int))
            best_val = best_m["macro_f1"]
            for th in np.linspace(-1.0, 1.0, 41):
                cur_m = evaluate_predictions(y_dev, (dec >= th).astype(int))
                if cur_m["macro_f1"] > best_val:
                    best_val = cur_m["macro_f1"]
                    best_th = float(th)
                    best_m = cur_m
            opt_th, opt_m = best_th, best_m

        default_preds = clf.predict(X_dev)
        default_m = evaluate_predictions(y_dev, default_preds)

        row = {
            "model": "LinearSVM",
            "params": str(cfg),
            "fit_time_sec": round(fit_dur, 2),
            "default_macro_f1": default_m["macro_f1"],
            "default_accuracy": default_m["accuracy"],
            "optimal_threshold": round(opt_th, 3),
            "tuned_macro_f1": opt_m["macro_f1"],
            "tuned_accuracy": opt_m["accuracy"],
            "tuned_clickbait_f1": opt_m["clickbait_f1"],
            "tuned_clickbait_recall": opt_m["clickbait_recall"],
            "tuned_clickbait_precision": opt_m["clickbait_precision"],
        }
        records.append(row)

        if opt_m["macro_f1"] > best_lsvm_score:
            best_lsvm_score = opt_m["macro_f1"]
            best_lsvm_item = {
                "params": cfg,
                "optimal_threshold": round(opt_th, 3),
                "metrics": opt_m,
            }

    print(f"Best Linear SVM: Macro-F1 = {best_lsvm_score:.4f} (params={best_lsvm_item['params']}, th={best_lsvm_item['optimal_threshold']})")
    best_configs["LinearSVM"] = best_lsvm_item

    # 3. XGBOOST TUNING
    print("\n--- Tuning XGBoost ---")
    xgb_grid = [
        {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 150, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 150, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"max_depth": 7, "learning_rate": 0.1, "n_estimators": 200, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"max_depth": 6, "learning_rate": 0.15, "n_estimators": 150, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 150, "colsample_bytree": 0.8, "scale_pos_weight": 1.25},
    ]

    best_xgb_score = -1.0
    best_xgb_item: dict[str, Any] = {}

    for cfg in xgb_grid:
        t_start = time.time()
        clf = XGBClassifier(
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=cfg["scale_pos_weight"],
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        fit_dur = time.time() - t_start

        probs = clf.predict_proba(X_dev)[:, 1]
        default_preds = (probs >= 0.5).astype(int)
        default_m = evaluate_predictions(y_dev, default_preds)

        opt_th, opt_m = tune_threshold(y_dev, probs)

        row = {
            "model": "XGBoost",
            "params": str(cfg),
            "fit_time_sec": round(fit_dur, 2),
            "default_macro_f1": default_m["macro_f1"],
            "default_accuracy": default_m["accuracy"],
            "optimal_threshold": round(opt_th, 3),
            "tuned_macro_f1": opt_m["macro_f1"],
            "tuned_accuracy": opt_m["accuracy"],
            "tuned_clickbait_f1": opt_m["clickbait_f1"],
            "tuned_clickbait_recall": opt_m["clickbait_recall"],
            "tuned_clickbait_precision": opt_m["clickbait_precision"],
        }
        records.append(row)

        if opt_m["macro_f1"] > best_xgb_score:
            best_xgb_score = opt_m["macro_f1"]
            best_xgb_item = {
                "params": cfg,
                "optimal_threshold": round(opt_th, 3),
                "metrics": opt_m,
            }

    print(f"Best XGBoost: Macro-F1 = {best_xgb_score:.4f} (params={best_xgb_item['params']}, th={best_xgb_item['optimal_threshold']})")
    best_configs["XGBoost"] = best_xgb_item

    # 4. RANDOM FOREST TUNING
    print("\n--- Tuning Random Forest ---")
    rf_grid = [
        {"n_estimators": 200, "max_depth": 35, "min_samples_split": 5, "class_weight": None},
        {"n_estimators": 200, "max_depth": 50, "min_samples_split": 5, "class_weight": None},
        {"n_estimators": 200, "max_depth": 70, "min_samples_split": 5, "class_weight": None},
        {"n_estimators": 250, "max_depth": 50, "min_samples_split": 5, "class_weight": "balanced"},
        {"n_estimators": 250, "max_depth": 60, "min_samples_split": 4, "class_weight": "balanced_subsample"},
    ]

    best_rf_score = -1.0
    best_rf_item: dict[str, Any] = {}

    for cfg in rf_grid:
        t_start = time.time()
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            class_weight=cfg["class_weight"],
            random_state=args.seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        fit_dur = time.time() - t_start

        probs = clf.predict_proba(X_dev)[:, 1]
        default_preds = (probs >= 0.5).astype(int)
        default_m = evaluate_predictions(y_dev, default_preds)

        opt_th, opt_m = tune_threshold(y_dev, probs)

        row = {
            "model": "RandomForest",
            "params": str(cfg),
            "fit_time_sec": round(fit_dur, 2),
            "default_macro_f1": default_m["macro_f1"],
            "default_accuracy": default_m["accuracy"],
            "optimal_threshold": round(opt_th, 3),
            "tuned_macro_f1": opt_m["macro_f1"],
            "tuned_accuracy": opt_m["accuracy"],
            "tuned_clickbait_f1": opt_m["clickbait_f1"],
            "tuned_clickbait_recall": opt_m["clickbait_recall"],
            "tuned_clickbait_precision": opt_m["clickbait_precision"],
        }
        records.append(row)

        if opt_m["macro_f1"] > best_rf_score:
            best_rf_score = opt_m["macro_f1"]
            best_rf_item = {
                "params": cfg,
                "optimal_threshold": round(opt_th, 3),
                "metrics": opt_m,
            }

    print(f"Best Random Forest: Macro-F1 = {best_rf_score:.4f} (params={best_rf_item['params']}, th={best_rf_item['optimal_threshold']})")
    best_configs["RandomForest"] = best_rf_item

    # Save detailed CSV and best JSON
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(records)
    df_results.to_csv(args.out_csv, index=False)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed tuning results to {args.out_csv}")
    print(f"Saved best hyperparameters configuration to {args.out_json}")

    print("\nSummary of Best Tuned Models on Dev (Version 3):")
    for name, item in best_configs.items():
        m = item["metrics"]
        print(f"- {name}: Macro-F1 = {m['macro_f1']:.4f}, Accuracy = {m['accuracy']:.4f}, Clickbait-F1 = {m['clickbait_f1']:.4f} (th={item['optimal_threshold']})")


if __name__ == "__main__":
    main()
