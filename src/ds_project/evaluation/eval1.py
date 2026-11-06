"""Evaluation functions for human annotations and LLM predictions on DEV."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

VARIANTS = ("T", "TC", "TS", "TSC")


def to_bool(series: pd.Series) -> pd.Series:
    """Convert CSV true/false values to booleans."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def check_ids(df: pd.DataFrame, column: str, name: str) -> None:
    if column not in df.columns:
        raise ValueError(f"{name}: missing column {column}")
    if df[column].isna().any() or df[column].duplicated().any():
        raise ValueError(f"{name}: missing or duplicate {column}")


def check_gold(df_dev: pd.DataFrame, df_gold: pd.DataFrame):
    """Check DEV gold labels and find records without gold."""
    check_ids(df_dev, "record_id", "DEV")
    check_ids(df_gold, "record_id", "DEV gold")

    df_gold = df_gold.copy()
    df_gold["final_label"] = pd.to_numeric(df_gold["final_label"], errors="coerce")
    if not df_gold["final_label"].isin([0, 1]).all():
        raise ValueError("DEV gold contains labels other than 0 and 1")
    if not df_gold["record_id"].isin(df_dev["record_id"]).all():
        raise ValueError("DEV gold contains records outside DEV")

    df_gold["final_label"] = df_gold["final_label"].astype(int)
    df_missing = df_dev[~df_dev["record_id"].isin(df_gold["record_id"])].copy()
    summary = {
        "dev_n": len(df_dev),
        "gold_n": len(df_gold),
        "missing_gold": len(df_missing),
        "label_0": int((df_gold["final_label"] == 0).sum()),
        "label_1": int((df_gold["final_label"] == 1).sum()),
    }
    if "needs_adjudication" in df_gold.columns:
        summary["needs_adjudication"] = int(to_bool(df_gold["needs_adjudication"]).sum())
    return df_gold, df_missing, summary


def get_kappa(y_true, y_pred):
    # Kappa is undefined if both sides always use the same single class.
    if len(set(y_true) | set(y_pred)) == 1:
        return np.nan
    return cohen_kappa_score(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    """Binary classification metrics, with class 1 meaning clickbait."""
    if len(y_true) == 0:
        raise ValueError("No records to evaluate")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "kappa": get_kappa(y_true, y_pred),
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "precision_1": precision[1],
        "recall_1": recall[1],
        "f1_1": f1[1],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def analyze_human(df_adj: pd.DataFrame):
    """Agreement between the two original human annotations."""
    check_ids(df_adj, "annotation_id", "Adjudication log")
    df_pairs = df_adj.copy()
    df_pairs["label_a"] = pd.to_numeric(df_pairs["label_a"], errors="coerce")
    df_pairs["label_b"] = pd.to_numeric(df_pairs["label_b"], errors="coerce")
    df_pairs = df_pairs[
        df_pairs["label_a"].isin([0, 1]) & df_pairs["label_b"].isin([0, 1])
    ].copy()
    if df_pairs.empty:
        raise ValueError("No valid human annotation pairs")

    df_pairs["label_a"] = df_pairs["label_a"].astype(int)
    df_pairs["label_b"] = df_pairs["label_b"].astype(int)
    df_pairs["flag_a"] = to_bool(df_pairs["flag_a"])
    df_pairs["flag_b"] = to_bool(df_pairs["flag_b"])

    rows = []
    for (person_a, person_b), group in df_pairs.groupby(["annotator_a", "annotator_b"]):
        a = group["label_a"].to_numpy()
        b = group["label_b"].to_numpy()
        rows.append({
            "annotator_a": person_a,
            "annotator_b": person_b,
            "n": len(group),
            "raw_agreement": (a == b).mean(),
            "kappa": get_kappa(a, b),
            "disagreement_rate": (a != b).mean(),
            "flag_a_rate": group["flag_a"].mean(),
            "flag_b_rate": group["flag_b"].mean(),
        })

    a = df_pairs["label_a"].to_numpy()
    b = df_pairs["label_b"].to_numpy()
    adjudication = to_bool(df_adj["needs_adjudication"])
    summary = {
        "n_pairs": len(df_pairs),
        "raw_agreement": (a == b).mean(),
        "kappa": get_kappa(a, b),
        "adjudication_n": int(adjudication.sum()),
        "adjudication_rate": adjudication.mean(),
        "flag_a_rate": df_pairs["flag_a"].mean(),
        "flag_b_rate": df_pairs["flag_b"].mean(),
    }
    return pd.DataFrame(rows), summary


def prepare_llm(df_llm: pd.DataFrame, df_dev: pd.DataFrame,
                df_gold: pd.DataFrame, variant: str):
    """Keep usable LLM labels, match with human gold, and count coverage."""
    check_ids(df_llm, "record_id", variant)
    if not df_llm["record_id"].isin(df_dev["record_id"]).all():
        raise ValueError(f"{variant}: predictions contain records outside DEV")

    labels = pd.to_numeric(df_llm["label"], errors="coerce")
    usable = to_bool(df_llm["label_usable"]) if "label_usable" in df_llm else pd.Series(True, index=df_llm.index)
    valid = usable & labels.isin([0, 1])

    retry = (pd.to_numeric(df_llm["retry_count"], errors="coerce").fillna(0)
             if "retry_count" in df_llm else pd.Series(0, index=df_llm.index))
    coverage = {
        "variant": variant,
        "dev_n": len(df_dev),
        "returned_n": len(df_llm),
        "valid_n": int(valid.sum()),
        "invalid_n": int((~valid).sum()),
        "missing_n": len(df_dev) - len(df_llm),
        "coverage": valid.sum() / len(df_dev),
        "retry_record_n": int((retry > 0).sum()),
        "valid_no_retry_n": int((valid & (retry == 0)).sum()),
    }
    if "common_usable_without_fallback" in df_llm:
        coverage["common_no_fallback_n"] = int((
            valid & to_bool(df_llm["common_usable_without_fallback"])
        ).sum())

    df_pred = df_llm.loc[valid, ["record_id"]].copy()
    df_pred["pred"] = labels.loc[valid].astype(int)
    df_eval = df_gold[["record_id", "final_label"]].merge(
        df_pred, on="record_id", validate="one_to_one"
    )
    df_eval = df_eval.merge(df_dev, on="record_id", validate="one_to_one")
    coverage["evaluated_n"] = len(df_eval)
    return df_eval, coverage


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     n_bootstrap: int = 2000, seed: int = 2026):
    """95% CI of macro-F1(B) - macro-F1(A), on matching DEV records."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    df_common = df_a[["record_id", "final_label", "pred"]].merge(
        df_b[["record_id", "final_label", "pred"]],
        on="record_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    if df_common.empty:
        raise ValueError("No common records between the two variants")
    if not (df_common["final_label_a"] == df_common["final_label_b"]).all():
        raise ValueError("The two variants have different gold labels")

    y_true = df_common["final_label_a"].to_numpy(dtype=int)
    pred_a = df_common["pred_a"].to_numpy(dtype=int)
    pred_b = df_common["pred_b"].to_numpy(dtype=int)
    n = len(y_true)

    f1_a = f1_score(y_true, pred_a, labels=[0, 1], average="macro", zero_division=0)
    f1_b = f1_score(y_true, pred_b, labels=[0, 1], average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        score_a = f1_score(y_true[indices], pred_a[indices], labels=[0, 1],
                           average="macro", zero_division=0)
        score_b = f1_score(y_true[indices], pred_b[indices], labels=[0, 1],
                           average="macro", zero_division=0)
        differences.append(score_b - score_a)

    return {
        "n_common": n,
        "macro_f1_a": f1_a,
        "macro_f1_b": f1_b,
        "delta_macro_f1": f1_b - f1_a,
        "ci_low": np.percentile(differences, 2.5),
        "ci_high": np.percentile(differences, 97.5),
    }
