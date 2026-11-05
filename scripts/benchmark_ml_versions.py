"""Benchmark the three input variants (v1, v2, v3) on baseline ML models.

Models evaluated:
- Logistic Regression
- Linear SVM (LinearSVC / SGDClassifier)
- Random Forest
(and XGBoost / LightGBM if installed in the environment).

Usage:
    python scripts/benchmark_ml_versions.py
"""

from __future__ import annotations

from typing import Any

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.correction.loader import load_dev, load_train_set
from ds_project.features.pipeline import ClickbaitFeaturePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ML input variants.")
    parser.add_argument("--train-variant", type=str, default="full_corrected")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "reports/ml/ml_input_variants_benchmark.csv")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def get_available_classifiers(seed: int = 2026) -> dict[str, Any]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "LinearSVM": SGDClassifier(loss="hinge", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=seed),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=6, random_state=seed, eval_metric="logloss")
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=100, max_depth=6, random_state=seed)
    except ImportError:
        pass

    return models


def main() -> None:
    args = parse_args()

    print("Loading datasets...")
    df_train = load_train_set(variant=args.train_variant)
    df_dev = load_dev(include_gold=True)

    y_train = df_train["label"].to_numpy(dtype=int)
    y_dev = df_dev["label"].to_numpy(dtype=int)

    variants = [
        ("v1", "Version 1 (Baseline: TF-IDF Matrix)"),
        ("v2", "Version 2 (Extended: 25 Warsaw Stylometric Features)"),
        ("v3", "Version 3 (Combined: Concat TF-IDF + 25 Warsaw Features)"),
    ]

    results = []

    for var_code, var_desc in variants:
        print(f"Extracting features for {var_desc}...")
        t0 = time.time()
        pipeline = ClickbaitFeaturePipeline(variant=var_code)
        X_train = pipeline.fit_transform(df_train)
        X_dev = pipeline.transform(df_dev)
        prep_time = time.time() - t0

        classifiers = get_available_classifiers(seed=args.seed)
        for model_name, clf in classifiers.items():
            t1 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t1

            y_pred = clf.predict(X_dev)

            acc = accuracy_score(y_dev, y_pred)
            f1_macro = f1_score(y_dev, y_pred, average="macro")
            prec_macro = precision_score(y_dev, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_dev, y_pred, average="macro", zero_division=0)

            f1_clickbait = f1_score(y_dev, y_pred, pos_label=1, zero_division=0)

            results.append({
                "variant": var_code,
                "variant_description": var_desc,
                "features_dim": X_train.shape[1],
                "model": model_name,
                "accuracy": acc,
                "macro_f1": f1_macro,
                "macro_precision": prec_macro,
                "macro_recall": rec_macro,
                "clickbait_f1": f1_clickbait,
                "feature_prep_sec": round(prep_time, 2),
                "train_sec": round(train_time, 2),
            })

    df_res = pd.DataFrame(results)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(args.out_csv, index=False)

    print(f"\nSaved benchmark results to {args.out_csv}")
    print("\nBenchmark Summary (Dev Evaluation):")
    summary = df_res[["variant", "model", "features_dim", "accuracy", "macro_f1", "clickbait_f1"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
