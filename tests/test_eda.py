"""Unit tests for Exploratory Data Analysis routines."""

from __future__ import annotations

import unittest
from pathlib import Path
import pandas as pd

from ds_project.eda.analysis import (
    analyze_by_category,
    analyze_by_source,
    analyze_human_annotations,
    analyze_llm_variants,
    analyze_splits_and_sources,
    compute_text_stats,
)
from ds_project.eda.report import render_eda_markdown

ROOT = Path(__file__).resolve().parents[1]


class TestEdaAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = ROOT / "data/processed/dataset.csv"
        self.dev_gold_path = ROOT / "data/annotations/human/gold/dev_gold.csv"
        self.adj_path = ROOT / "data/annotations/human/dev/adjudication_log.csv"
        self.llm_dir = ROOT / "data/annotations/llm"

        self.df_dataset = pd.read_csv(self.dataset_path)

    def test_text_stats(self) -> None:
        stats = compute_text_stats(self.df_dataset["title"])
        self.assertIn("char_mean", stats)
        self.assertIn("word_mean", stats)
        self.assertGreater(stats["char_mean"], 30.0)
        self.assertGreater(stats["word_mean"], 5.0)

    def test_splits_and_sources(self) -> None:
        table = analyze_splits_and_sources(self.df_dataset)
        self.assertIn("Total", table.columns)
        total_val = int(table[table["source"] == "Total"]["Total"].iloc[0])
        self.assertEqual(total_val, 23600)

    def test_by_category_and_source(self) -> None:
        df_cat = analyze_by_category(self.df_dataset)
        self.assertEqual(len(df_cat), 16)
        self.assertAlmostEqual(df_cat["share"].sum(), 1.0, places=4)

        df_src = analyze_by_source(self.df_dataset)
        self.assertEqual(len(df_src), 12)
        self.assertAlmostEqual(df_src["share"].sum(), 1.0, places=4)

    def test_human_annotations_agreement(self) -> None:
        h_stats = analyze_human_annotations(self.dev_gold_path, self.adj_path)
        self.assertEqual(h_stats["total_dev_gold"], 1395)
        self.assertGreater(h_stats["raw_agreement_rate"], 0.70)
        self.assertGreater(h_stats["cohen_kappa"], 0.40)
        self.assertGreater(h_stats["adjudication_rate"], 0.15)

    def test_llm_variants(self) -> None:
        df_variants, df_disagree = analyze_llm_variants(
            self.dev_gold_path,
            self.llm_dir,
            self.df_dataset,
        )
        self.assertEqual(len(df_variants), 4)
        for _, row in df_variants.iterrows():
            self.assertGreater(row["accuracy"], 0.70)
            self.assertGreater(row["macro_f1"], 0.70)

        self.assertFalse(df_disagree.empty)


if __name__ == "__main__":
    unittest.main()
