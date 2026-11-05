"""Core analytical routines for Exploratory Data Analysis (EDA).

Analyzes:
- Dataset volume, splits, sources, categories, outlet groups
- Title length distributions and missing sapo rates
- Rejection and data issue summaries
- Human annotator agreement (Cohen's Kappa, adjudication rate) on dev
- LLM performance and disagreement on dev across 4 input variants (T, TC, TS, TSC)
- Disagreement breakdown by source and category
- Train set label distribution and concentration across sources/categories
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support


def compute_text_stats(series: pd.Series) -> dict[str, float]:
    """Compute descriptive statistics for text lengths."""
    lengths = series.fillna("").astype(str).str.len()
    words = series.fillna("").astype(str).str.split().str.len()

    return {
        "char_mean": float(lengths.mean()),
        "char_std": float(lengths.std()),
        "char_median": float(lengths.median()),
        "char_min": float(lengths.min()),
        "char_max": float(lengths.max()),
        "char_p25": float(lengths.quantile(0.25)),
        "char_p75": float(lengths.quantile(0.75)),
        "word_mean": float(words.mean()),
        "word_std": float(words.std()),
        "word_median": float(words.median()),
        "word_min": float(words.min()),
        "word_max": float(words.max()),
        "word_p25": float(words.quantile(0.25)),
        "word_p75": float(words.quantile(0.75)),
    }


def analyze_splits_and_sources(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate records by source, outlet group, and split."""
    table = pd.crosstab(
        index=[df_dataset["source"], df_dataset["outlet_group"]],
        columns=df_dataset["split"],
        margins=True,
        margins_name="Total",
    ).reset_index()
    return table


def analyze_by_category(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize records, missing sapo rate, and title word length per category."""
    records = []
    for cat, group in df_dataset.groupby("category", dropna=False):
        n = len(group)
        missing_sapo = group["sapo"].isna().sum()
        sapo_pct = float(missing_sapo / n) if n else 0.0
        word_lens = group["title"].fillna("").astype(str).str.split().str.len()
        records.append({
            "category": str(cat),
            "records": n,
            "share": float(n / len(df_dataset)),
            "missing_sapo_count": int(missing_sapo),
            "missing_sapo_pct": sapo_pct,
            "title_word_mean": float(word_lens.mean()),
            "title_word_median": float(word_lens.median()),
        })
    df_cat = pd.DataFrame(records).sort_values(by="records", ascending=False)
    return df_cat


def analyze_by_source(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize records, missing sapo rate, and title word length per source."""
    records = []
    for (src, grp_name), group in df_dataset.groupby(["source", "outlet_group"], dropna=False):
        n = len(group)
        missing_sapo = group["sapo"].isna().sum()
        sapo_pct = float(missing_sapo / n) if n else 0.0
        word_lens = group["title"].fillna("").astype(str).str.split().str.len()
        records.append({
            "source": str(src),
            "outlet_group": str(grp_name),
            "records": n,
            "share": float(n / len(df_dataset)),
            "missing_sapo_count": int(missing_sapo),
            "missing_sapo_pct": sapo_pct,
            "title_word_mean": float(word_lens.mean()),
            "title_word_median": float(word_lens.median()),
        })
    df_src = pd.DataFrame(records).sort_values(by="records", ascending=False)
    return df_src


def analyze_human_annotations(
    dev_gold_path: Path,
    adjudication_log_path: Path,
) -> dict[str, Any]:
    """Calculate inter-annotator agreement and human gold statistics on dev."""
    df_gold = pd.read_csv(dev_gold_path)
    df_adj = pd.read_csv(adjudication_log_path)

    # Gold label distribution
    gold_counts = df_gold["final_label"].value_counts(dropna=False).to_dict()
    total_gold = len(df_gold)
    gold_pcts = {k: v / total_gold for k, v in gold_counts.items()}

    # Adjudication rate
    adj_needed = df_adj["needs_adjudication"].sum()
    adj_rate = float(adj_needed / len(df_adj)) if len(df_adj) else 0.0

    # Annotator agreement (label_a vs label_b)
    valid_pair = df_adj.dropna(subset=["label_a", "label_b"]).copy()
    valid_pair["label_a"] = valid_pair["label_a"].astype(int)
    valid_pair["label_b"] = valid_pair["label_b"].astype(int)

    raw_agree = float((valid_pair["label_a"] == valid_pair["label_b"]).mean())
    kappa = float(cohen_kappa_score(valid_pair["label_a"], valid_pair["label_b"]))

    # Flag rate
    flag_a_rate = float(valid_pair["flag_a"].mean()) if "flag_a" in valid_pair.columns else 0.0
    flag_b_rate = float(valid_pair["flag_b"].mean()) if "flag_b" in valid_pair.columns else 0.0

    return {
        "total_dev_gold": total_gold,
        "gold_counts": gold_counts,
        "gold_pcts": gold_pcts,
        "adjudication_needed_count": int(adj_needed),
        "adjudication_rate": adj_rate,
        "raw_agreement_rate": raw_agree,
        "cohen_kappa": kappa,
        "annotator_a_flag_rate": flag_a_rate,
        "annotator_b_flag_rate": flag_b_rate,
    }


def analyze_llm_variants(
    dev_gold_path: Path,
    llm_dir: Path,
    df_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate LLM variants on dev against human gold and compute breakdown."""
    df_gold = pd.read_csv(dev_gold_path).dropna(subset=["record_id", "final_label"])
    df_gold["final_label"] = df_gold["final_label"].astype(int)

    variants = [
        ("T (Title only)", llm_dir / "llm_dev_T.csv"),
        ("TC (Title + Category)", llm_dir / "llm_dev_TC.csv"),
        ("TS (Title + Sapo)", llm_dir / "llm_dev_TS.csv"),
        ("TSC (Title + Sapo + Category)", llm_dir / "llm_dev_TSC.csv"),
    ]

    variant_metrics = []
    disagreement_records = []

    dev_meta = df_dataset[df_dataset["split"] == "dev"][["record_id", "source", "category"]].copy()

    for name, path in variants:
        if not path.exists():
            continue
        df_llm = pd.read_csv(path)
        # Drop redundant metadata columns from df_llm if already merging with dev_meta
        cols_to_use = [c for c in df_llm.columns if c not in ("source", "category")]
        m = pd.merge(df_gold, df_llm[cols_to_use], on="record_id")
        m = pd.merge(m, dev_meta, on="record_id", how="left")
        m = m.dropna(subset=["final_label", "label"])
        m["label"] = m["label"].astype(int)

        y_true = m["final_label"].to_numpy()
        y_pred = m["label"].to_numpy()

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        acc = float((y_true == y_pred).mean())

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        variant_metrics.append({
            "variant": name,
            "evaluated_n": len(m),
            "accuracy": acc,
            "macro_f1": float(f1),
            "macro_precision": float(p),
            "macro_recall": float(r),
            "fp_0_to_1": int(fp),
            "fn_1_to_0": int(fn),
            "disagreement_rate": float((fp + fn) / len(m)),
        })

        # Breakdown by source and category for TSC (default chosen configuration)
        if "TSC" in name:
            for src, grp in m.groupby("source"):
                sub_true = grp["final_label"].to_numpy()
                sub_pred = grp["label"].to_numpy()
                fp_s = int(((sub_true == 0) & (sub_pred == 1)).sum())
                fn_s = int(((sub_true == 1) & (sub_pred == 0)).sum())
                disagreement_records.append({
                    "dimension": "source",
                    "group": src,
                    "records": len(grp),
                    "fp_0_to_1": fp_s,
                    "fn_1_to_0": fn_s,
                    "disagreements": fp_s + fn_s,
                    "disagreement_rate": float((fp_s + fn_s) / len(grp)),
                })
            for cat, grp in m.groupby("category"):
                sub_true = grp["final_label"].to_numpy()
                sub_pred = grp["label"].to_numpy()
                fp_c = int(((sub_true == 0) & (sub_pred == 1)).sum())
                fn_c = int(((sub_true == 1) & (sub_pred == 0)).sum())
                disagreement_records.append({
                    "dimension": "category",
                    "group": cat,
                    "records": len(grp),
                    "fp_0_to_1": fp_c,
                    "fn_1_to_0": fn_c,
                    "disagreements": fp_c + fn_c,
                    "disagreement_rate": float((fp_c + fn_c) / len(grp)),
                })

    df_variant_metrics = pd.DataFrame(variant_metrics)
    df_disagreement = pd.DataFrame(disagreement_records)
    return df_variant_metrics, df_disagreement


def analyze_train_labels(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze label distribution across sources and categories in the training set."""
    label_col = "label" if "label" in df_train.columns else "llm_label"

    # Per source
    src_rows = []
    for src, grp in df_train.groupby("source"):
        n = len(grp)
        cb = int((grp[label_col] == 1).sum())
        src_rows.append({
            "source": src,
            "records": n,
            "clickbait_count": cb,
            "clickbait_ratio": float(cb / n) if n else 0.0,
            "non_clickbait_count": n - cb,
        })
    df_src_labels = pd.DataFrame(src_rows).sort_values(by="clickbait_ratio", ascending=False)

    # Per category
    cat_rows = []
    for cat, grp in df_train.groupby("category"):
        n = len(grp)
        cb = int((grp[label_col] == 1).sum())
        cat_rows.append({
            "category": cat,
            "records": n,
            "clickbait_count": cb,
            "clickbait_ratio": float(cb / n) if n else 0.0,
            "non_clickbait_count": n - cb,
        })
    df_cat_labels = pd.DataFrame(cat_rows).sort_values(by="clickbait_ratio", ascending=False)

    return df_src_labels, df_cat_labels
