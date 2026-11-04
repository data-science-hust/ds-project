"""Exploratory Data Analysis (EDA) package."""

from ds_project.eda.analysis import (
    analyze_by_category,
    analyze_by_source,
    analyze_human_annotations,
    analyze_llm_variants,
    analyze_splits_and_sources,
    analyze_train_labels,
    compute_text_stats,
)
from ds_project.eda.report import render_eda_markdown

__all__ = [
    "compute_text_stats",
    "analyze_splits_and_sources",
    "analyze_by_category",
    "analyze_by_source",
    "analyze_human_annotations",
    "analyze_llm_variants",
    "analyze_train_labels",
    "render_eda_markdown",
]
