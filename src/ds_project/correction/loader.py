"""Unified data and label loader for machine learning models.

Provides standardized access to full train, correction ablation subsets,
dev set with human gold, and test sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ds_project.features.pipeline import ClickbaitFeaturePipeline, InputVariant

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


def load_dataset(dataset_path: Path | str | None = None) -> pd.DataFrame:
    """Load the master dataset table (23,600 records)."""
    p = Path(dataset_path) if dataset_path else PROCESSED_DIR / "dataset.csv"
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found at {p}")
    return pd.read_csv(p)


def load_train_set(
    variant: Literal["full_corrected", "llm_raw"] = "full_corrected",
    train_corrected_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the full 20,000 training records.

    Parameters
    ----------
    variant : {"full_corrected", "llm_raw"}
        - "full_corrected": uses human gold on the 6,400 subset and LLM elsewhere.
        - "llm_raw": uses LLM labels across all 20,000 records.
    """
    p = Path(train_corrected_path) if train_corrected_path else PROCESSED_DIR / "train_full_corrected.csv"
    if p.exists():
        df = pd.read_csv(p)
        if variant == "llm_raw" and "llm_label" in df.columns:
            df["label"] = df["llm_label"]
        return df

    # Fallback to merging dataset.csv with llm_train.csv
    df_dataset = load_dataset()
    df_train = df_dataset[df_dataset["split"] == "train"].copy()
    llm_path = ANNOTATIONS_DIR / "llm" / "llm_train.csv"
    if llm_path.exists():
        df_llm = pd.read_csv(llm_path)
        llm_map = df_llm.set_index("record_id")["label"].fillna(0).astype(int).to_dict()
        df_train["label"] = [llm_map.get(rid, 0) for rid in df_train["record_id"]]
    else:
        df_train["label"] = 0

    return df_train


def load_subset(
    correction_level: float = 1.0,
    seed: int = 2026,
    subset_versions_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the 6,400 training subset records with labels at a specific correction level.

    Parameters
    ----------
    correction_level : float
        Correction fraction in {0.0, 0.25, 0.50, 0.75, 1.0}.
    seed : int
        Seed for the nested ablation subset (e.g. 2026, 42, 123).
    """
    df_dataset = load_dataset()
    df_subset = df_dataset[
        (df_dataset["split"] == "train") & (df_dataset["is_human_subset"] == True)
    ].copy()

    p = Path(subset_versions_path) if subset_versions_path else MANIFESTS_DIR / "subset_label_versions.csv"
    if p.exists():
        df_vers = pd.read_csv(p)
        matched = df_vers[
            (df_vers["seed"] == seed) & (np.isclose(df_vers["correction_level"], correction_level))
        ]
        if not matched.empty:
            label_map = matched.set_index("record_id")["label"].to_dict()
            df_subset["label"] = [label_map.get(rid, 0) for rid in df_subset["record_id"]]
            return df_subset

    # Fallback: load full corrected train and filter
    df_train = load_train_set(variant="full_corrected")
    return df_train[df_train["is_human_subset"] == True].copy()


def load_dev(include_gold: bool = True) -> pd.DataFrame:
    """Load the 1,400 dev set records with human gold labels."""
    df_dataset = load_dataset()
    df_dev = df_dataset[df_dataset["split"] == "dev"].copy()

    if include_gold:
        gold_path = ANNOTATIONS_DIR / "human" / "gold" / "dev_gold.csv"
        if gold_path.exists():
            df_gold = pd.read_csv(gold_path)
            gold_map = df_gold.dropna(subset=["record_id", "final_label"]).set_index("record_id")["final_label"].astype(int).to_dict()
            df_dev["label"] = [gold_map.get(rid, np.nan) for rid in df_dev["record_id"]]
            # Filter unadjudicated / excluded rows if any
            df_dev = df_dev.dropna(subset=["label"]).copy()
            df_dev["label"] = df_dev["label"].astype(int)

    return df_dev


def load_test(kind: Literal["seen", "unseen"] = "seen") -> pd.DataFrame:
    """Load test records (1,000 for seen, 1,200 for unseen)."""
    split_name = "test_seen" if kind == "seen" else "test_unseen"
    df_dataset = load_dataset()
    return df_dataset[df_dataset["split"] == split_name].copy()


def get_train_dev_matrices(
    train_variant: Literal["full_corrected", "llm_raw"] = "full_corrected",
    input_variant: InputVariant = "v1",
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, ClickbaitFeaturePipeline]:
    """Convenience helper returning fitted matrices (X_train, y_train, X_dev, y_dev, pipeline)."""
    df_train = load_train_set(variant=train_variant)
    df_dev = load_dev(include_gold=True)

    pipeline = ClickbaitFeaturePipeline(variant=input_variant)
    X_train = pipeline.fit_transform(df_train)
    y_train = df_train["label"].to_numpy(dtype=int)

    X_dev = pipeline.transform(df_dev)
    y_dev = df_dev["label"].to_numpy(dtype=int)

    return X_train, y_train, X_dev, y_dev, pipeline
