"""Unit tests for the 25 Warsaw stylometric features extractor."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from ds_project.features.stylometric import (
    FEATURE_NAMES,
    WarsawStylometricExtractor,
    extract_headline_features,
)


class TestStylometricFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = WarsawStylometricExtractor()

    def test_feature_count_and_names(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 25)
        self.assertEqual(len(self.extractor.get_feature_names_out()), 25)

    def test_single_headline_extraction(self) -> None:
        title = "Sự thật bất ngờ về mức lương khủng của HLV Park: Tiết lộ gây sốc!"
        sapo = "HLV Park Hang Seo nhận mức thù lao lớn từ VFF trong hợp đồng mới."
        feats = extract_headline_features(title, sapo)

        self.assertEqual(len(feats), 25)
        self.assertGreater(feats["char_count"], 0)
        self.assertGreater(feats["word_count"], 0)
        self.assertGreater(feats["bait_punctuation_count"], 0)  # Has '!' and ':'
        self.assertGreater(feats["similarity_score"], 0)  # Shares terms with sapo
        self.assertGreater(feats["bait_phrases_count"], 0)  # Contains 'sự thật', 'tiết lộ', 'sốc'

        # Check all values are finite
        for k, v in feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} is not finite: {v}")

    def test_edge_cases(self) -> None:
        # Empty string
        empty_feats = extract_headline_features("", None)
        self.assertEqual(len(empty_feats), 25)
        for k, v in empty_feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} on empty string is not finite: {v}")

        # Single word
        single_feats = extract_headline_features("Tin", None)
        self.assertEqual(len(single_feats), 25)
        for k, v in single_feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} on single word is not finite: {v}")

    def test_batch_transform(self) -> None:
        df = pd.DataFrame({
            "title": [
                "Hà Nội công bố điểm chuẩn vào lớp 10 năm 2026",
                "Bật mí danh tính người đẹp hẹn hò với nam ca sĩ đình đám?",
                "Giá xăng dầu hôm nay tăng nhẹ",
            ],
            "sapo": [
                "Sở Giáo dục và Đào tạo Hà Nội thông báo điểm chuẩn...",
                "Cộng đồng mạng xôn xao bàn tán...",
                None,
            ],
        })

        mat = self.extractor.transform(df)
        self.assertEqual(mat.shape, (3, 25))
        self.assertFalse(np.isnan(mat).any())
        self.assertFalse(np.isinf(mat).any())

        df_out = self.extractor.transform_df(df)
        self.assertEqual(list(df_out.columns), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
