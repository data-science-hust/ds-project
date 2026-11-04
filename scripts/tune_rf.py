"""Tune Random Forest hyperparameters and decision threshold on the Dev set.

Appends results to reports/hyperparameter_tuning_results.csv and updates
reports/best_hyperparameters.json.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.correction.loader import load_dev, load_train_set
from ds_project.features.pipeline import ClickbaitFeaturePipeline


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


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    best_th = 0.5
    best_metrics: dict[str, float] = {}
    best_f1 = -1.0

    thresholds = np.linspace(0.25, 0.65, 41)
    for th in thresholds:
        preds = (scores >= th).astype(int)
        m = evaluate_predictions(y_true, preds)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_th = float(th)
            best_metrics = m

    return best_th, best_metrics


def main() -> None:
    seed = 2026
    csv_path = ROOT / "reports/hyperparameter_tuning_results.csv"
    json_path = ROOT / "reports/best_hyperparameters.json"

    print("Loading datasets for Random Forest tuning...")
    df_train = load_train_set(variant="full_corrected")
    df_dev = load_dev(include_gold=True)

    y_train = df_train["label"].to_numpy(dtype=int)
    y_dev = df_dev["label"].to_numpy(dtype=int)

    print("Extracting features (v3)...")
    pipeline = ClickbaitFeaturePipeline(variant="v3")
    X_train = pipeline.fit_transform(df_train)
    X_dev = pipeline.transform(df_dev)

    rf_grid = [
        {"n_estimators": 200, "max_depth": 35, "min_samples_split": 5, "class_weight": None},
        {"n_estimators": 200, "max_depth": 50, "min_samples_split": 5, "class_weight": None},
        {"n_estimators": 200, "max_depth": 70, "min_samples_split": 5, "class_weight": None},
        {"n_estimators": 250, "max_depth": 50, "min_samples_split": 5, "class_weight": "balanced"},
        {"n_estimators": 250, "max_depth": 60, "min_samples_split": 4, "class_weight": "balanced_subsample"},
    ]

    records: list[dict[str, Any]] = []
    best_rf_score = -1.0
    best_rf_item: dict[str, Any] = {}

    print("\n--- Tuning Random Forest ---")
    for cfg in rf_grid:
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            class_weight=cfg["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        fit_dur = time.time() - t0

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
        print(f"Config: {cfg} -> Default Macro-F1: {default_m['macro_f1']:.4f}, Tuned Macro-F1: {opt_m['macro_f1']:.4f} (th={opt_th:.2f}, time={fit_dur:.1f}s)")

        if opt_m["macro_f1"] > best_rf_score:
            best_rf_score = opt_m["macro_f1"]
            best_rf_item = {
                "params": cfg,
                "optimal_threshold": round(opt_th, 3),
                "metrics": opt_m,
            }

    print(f"\nBest Random Forest: Macro-F1 = {best_rf_score:.4f} (th={best_rf_item['optimal_threshold']})")

    # Update CSV
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        # Drop existing RandomForest if any
        df_old = df_old[df_old["model"] != "RandomForest"]
        df_new = pd.concat([df_old, pd.DataFrame(records)], ignore_index=True)
    else:
        df_new = pd.DataFrame(records)
    df_new.to_csv(csv_path, index=False)

    # Update JSON
    best_dict = {}
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            best_dict = json.load(f)
    best_dict["RandomForest"] = best_rf_item
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(best_dict, f, indent=2, ensure_ascii=False)

    print(f"Updated {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
