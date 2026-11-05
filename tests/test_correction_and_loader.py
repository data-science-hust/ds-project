"""Unit tests for label correction, nested ablation generator, and dataset loader."""

from __future__ import annotations

import unittest
import pandas as pd

from ds_project.correction.correction import (
    DEFAULT_CORRECTION_LEVELS,
    DEFAULT_SEEDS,
    build_full_train_after_correction,
    build_nested_subset_corrections,
)
from ds_project.correction.loader import (
    load_dataset,
    load_dev,
    load_subset,
    load_test,
    load_train_set,
)


class TestCorrectionAndLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.df_dataset = load_dataset()

    def test_load_dataset(self) -> None:
        self.assertEqual(len(self.df_dataset), 23600)
        self.assertIn("record_id", self.df_dataset.columns)
        self.assertIn("split", self.df_dataset.columns)
        self.assertIn("is_human_subset", self.df_dataset.columns)

    def test_load_splits(self) -> None:
        dev = load_dev(include_gold=True)
        self.assertEqual(len(dev), 1395)
        self.assertTrue(set(dev["label"].unique()).issubset({0, 1}))

        test_seen = load_test("seen")
        self.assertEqual(len(test_seen), 1000)

        test_unseen = load_test("unseen")
        self.assertEqual(len(test_unseen), 1200)

    def test_load_train_set_variants(self) -> None:
        df_full = load_train_set("full_corrected")
        self.assertEqual(len(df_full), 20000)
        self.assertIn("label", df_full.columns)
        self.assertTrue(set(df_full["label"].unique()).issubset({0, 1}))

        df_llm = load_train_set("llm_raw")
        self.assertEqual(len(df_llm), 20000)
        self.assertIn("label", df_llm.columns)

    def test_nested_correction_property(self) -> None:
        # Dummy records to verify nesting
        subset_recs = pd.DataFrame({
            "record_id": [f"rec_{i}" for i in range(100)],
            "is_human_subset": [True] * 100,
        })
        # 50 agreements, 50 disagreements
        llm_df = pd.DataFrame({
            "record_id": [f"rec_{i}" for i in range(100)],
            "label": [0] * 100,
        })
        gold_df = pd.DataFrame({
            "record_id": [f"rec_{i}" for i in range(100)],
            "final_label": [0] * 50 + [1] * 50,
        })

        manifest_df, versions_df, stats = build_nested_subset_corrections(
            df_subset_records=subset_recs,
            df_llm_subset=llm_df,
            df_train_gold=gold_df,
            levels=(0.0, 0.25, 0.50, 0.75, 1.0),
            seeds=(2026, 42),
        )

        for s in (2026, 42):
            s_manifest = manifest_df[manifest_df["seed"] == s]
            corrected_0 = set(s_manifest[(s_manifest["correction_level"] == 0.0) & (s_manifest["was_corrected"])]["record_id"])
            corrected_25 = set(s_manifest[(s_manifest["correction_level"] == 0.25) & (s_manifest["was_corrected"])]["record_id"])
            corrected_50 = set(s_manifest[(s_manifest["correction_level"] == 0.50) & (s_manifest["was_corrected"])]["record_id"])
            corrected_75 = set(s_manifest[(s_manifest["correction_level"] == 0.75) & (s_manifest["was_corrected"])]["record_id"])
            corrected_100 = set(s_manifest[(s_manifest["correction_level"] == 1.00) & (s_manifest["was_corrected"])]["record_id"])

            self.assertEqual(len(corrected_0), 0)
            self.assertEqual(len(corrected_25), 12)
            self.assertEqual(len(corrected_50), 25)
            self.assertEqual(len(corrected_75), 38)
            self.assertEqual(len(corrected_100), 50)

            # Strict nesting verification
            self.assertTrue(corrected_0.issubset(corrected_25))
            self.assertTrue(corrected_25.issubset(corrected_50))
            self.assertTrue(corrected_50.issubset(corrected_75))
            self.assertTrue(corrected_75.issubset(corrected_100))

    def test_load_subset_ablation(self) -> None:
        sub_0 = load_subset(correction_level=0.0, seed=2026)
        sub_100 = load_subset(correction_level=1.0, seed=2026)

        self.assertEqual(len(sub_0), 6400)
        self.assertEqual(len(sub_100), 6400)
        # Disagreements should be corrected between level 0.0 and level 1.0
        self.assertGreater((sub_0["label"] != sub_100["label"]).sum(), 0)


if __name__ == "__main__":
    unittest.main()
