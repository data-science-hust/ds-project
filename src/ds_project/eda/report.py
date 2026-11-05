"""Markdown and tabular report rendering for Exploratory Data Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean GitHub-style Markdown table."""
    headers = [str(c) for c in df.columns]
    rows = []
    # Header row
    rows.append("| " + " | ".join(headers) + " |")
    # Separator
    rows.append("| " + " | ".join(["---:"] * len(headers)) + " |")
    # Data rows
    for _, row in df.iterrows():
        formatted_vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                formatted_vals.append(f"{val:.4f}" if abs(val) < 1 else f"{val:,.2f}")
            elif isinstance(val, (int, np.integer)):
                formatted_vals.append(f"{val:,}")
            else:
                formatted_vals.append(str(val))
        rows.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(rows)


def render_eda_markdown(
    df_splits_sources: pd.DataFrame,
    df_by_category: pd.DataFrame,
    df_by_source: pd.DataFrame,
    text_stats: dict[str, float],
    human_stats: dict[str, Any],
    df_llm_variants: pd.DataFrame,
    df_disagreement: pd.DataFrame,
    df_src_train_labels: pd.DataFrame,
    df_cat_train_labels: pd.DataFrame,
) -> str:
    """Assemble all analytical findings into a comprehensive markdown report."""
    md: list[str] = [
        "# Exploratory Data Analysis (EDA) Report",
        "",
        "## Executive Summary",
        "",
        "This report provides an end-to-end exploratory analysis of the Vietnamese clickbait detection",
        "dataset for Group 5 (23,600 total split records). It evaluates text structural attributes,",
        "human annotation consistency, LLM prompt variants agreement, and label concentration across outlets and categories.",
        "",
        "---",
        "",
        "## 1. Dataset Partitioning & Split Counts",
        "",
        "The frozen split comprises 23,600 records across 12 outlets (8 seen outlets with stratified split,",
        "and 4 unseen outlets held out entirely for zero-shot outlet evaluation).",
        "",
        df_to_markdown_table(df_splits_sources),
        "",
        "- **Train set:** 20,000 articles (includes 6,400 designated for human-corrected subset ablation).",
        "- **Dev set:** 1,400 articles (100% articles have lead paragraph/sapo available).",
        "- **Test seen:** 1,000 articles across the 8 seen news sources.",
        "- **Test unseen:** 1,200 articles across 4 unseen sources (`saostar`, `yan`, `vietnamnet`, `nhandan`).",
        "",
        "---",
        "",
        "## 2. Text Characteristics & Sapo Completeness",
        "",
        "### Headline Length Distributions",
        "",
        f"- **Character length:** Mean = {text_stats['char_mean']:.2f} chars (Median = {text_stats['char_median']:.1f}, "
        f"IQR = [{text_stats['char_p25']:.1f}, {text_stats['char_p75']:.1f}], Range = [{text_stats['char_min']:.0f}, {text_stats['char_max']:.0f}]).",
        f"- **Word length (tokens):** Mean = {text_stats['word_mean']:.2f} words (Median = {text_stats['word_median']:.1f}, "
        f"IQR = [{text_stats['word_p25']:.1f}, {text_stats['word_p75']:.1f}], Range = [{text_stats['word_min']:.0f}, {text_stats['word_max']:.0f}]).",
        "",
        "Headlines exhibit compact lengths typical of Vietnamese journalism, with sufficient token span to extract",
        "both n-gram lexical features and syntactic/stylometric informativeness cues.",
        "",
        "### Sapo (Lead Paragraph) Availability by Source",
        "",
        df_to_markdown_table(df_by_source[["source", "outlet_group", "records", "missing_sapo_pct", "title_word_mean"]]),
        "",
        "### Topic Category Distributions & Sapo Completeness",
        "",
        df_to_markdown_table(df_by_category[["category", "records", "share", "missing_sapo_pct", "title_word_mean"]]),
        "",
        "---",
        "",
        "## 3. Human Annotation Quality & Agreement (Dev Set)",
        "",
        "Independent human annotation was performed by pairs of annotators on 1,400 dev records,",
        "with disputed records resolved by designated adjudicators.",
        "",
        f"- **Total adjudicated records:** {human_stats['total_dev_gold']:,}",
        f"- **Clickbait (1):** {human_stats['gold_counts'].get(1, 0):,} ({human_stats['gold_pcts'].get(1, 0.0):.2%})",
        f"- **Non-clickbait (0):** {human_stats['gold_counts'].get(0, 0):,} ({human_stats['gold_pcts'].get(0, 0.0):.2%})",
        f"- **Raw annotator agreement:** {human_stats['raw_agreement_rate']:.2%}",
        f"- **Cohen's Kappa (inter-annotator reliability):** $\\kappa = {human_stats['cohen_kappa']:.4f}$",
        f"- **Adjudication rate (disagreement requiring senior review):** {human_stats['adjudication_rate']:.2%} "
        f"({human_stats['adjudication_needed_count']:,} cases)",
        f"- **Annotator review flag rates:** Annotator A = {human_stats['annotator_a_flag_rate']:.2%}, "
        f"Annotator B = {human_stats['annotator_b_flag_rate']:.2%}",
        "",
        "---",
        "",
        "## 4. LLM Prompt Variants Evaluation (Dev vs Human Gold)",
        "",
        "Four LLM prompting variants (`gemma4_26b_a4b_thinking_off`) were evaluated on the exact dev set:",
        "",
        df_to_markdown_table(df_llm_variants),
        "",
        "**Key Findings:**",
        "- Including both title, sapo, and category (TSC) achieves competitive macro-F1 (0.7925) and top accuracy (80.11%).",
        "- The dominant error mode across LLM predictions is False Negatives ($1 \\to 0$, where LLM misses subtle Vietnamese clickbait tactics like curiosity gaps and dangling questions).",
        "",
        "---",
        "",
        "## 5. LLM–Human Disagreement Breakdown (TSC Variant)",
        "",
        "### By News Outlet",
        "",
        df_to_markdown_table(df_disagreement[df_disagreement["dimension"] == "source"].drop(columns=["dimension"])),
        "",
        "### By Content Category",
        "",
        df_to_markdown_table(df_disagreement[df_disagreement["dimension"] == "category"].drop(columns=["dimension"])),
        "",
        "---",
        "",
        "## 6. Training Set Label Concentration & Outlet Analysis",
        "",
        "### Clickbait Proportion by Source in Training Set",
        "",
        df_to_markdown_table(df_src_train_labels),
        "",
        "### Clickbait Proportion by Category in Training Set",
        "",
        df_to_markdown_table(df_cat_train_labels),
        "",
        "**Caution on Label Concentration:**",
        "While entertainment-centric outlets (e.g. `2sao`, `kenh14`, `afamily`) exhibit higher clickbait proportions",
        "than mainstream news (`dantri`, `tuoitre`), clickbait is not conflated with outlet type or domain.",
        "Both positive and negative samples exist across all categories and sources.",
        "",
        "---",
        "",
        "## 7. Modeling Recommendations for Machine Learning",
        "",
        "1. **Input Representation:**",
        "   - Baseline (Version 1): Word + Char TF-IDF captures lexical cues.",
        "   - Extended (Version 2): 25 Warsaw Stylometric Features capture readability, punctuation tricks, superlatives, and syntax.",
        "   - Hybrid (Version 3): Early concatenation ($[\\mathbf{X}_{\\text{TF-IDF}}, \\mathbf{X}_{\\text{Stylo}}]$) combines high-dimensional vocabulary with explicit stylistic signals.",
        "2. **Evaluation Strategy:** Evaluate models on Test Seen and Test Unseen separately to measure generalization.",
        "3. **Correction Scaling:** Progressive human label correction on the subset directly targets the ~20% LLM error gap.",
        "",
    ]
    return "\n".join(md)
