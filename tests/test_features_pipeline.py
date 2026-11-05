"""Unit tests for the multi-variant Feature Pipeline (Version 1, 2, and 3)."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression

from ds_project.features.pipeline import ClickbaitFeaturePipeline


class TestClickbaitFeaturePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.df_train = pd.DataFrame({
            "title": [
                "Bật mí 5 bí mật động trời về vụ án mạng",
                "Chính phủ ban hành nghị định mới về thuế thu nhập",
                "Cái kết bất ngờ cho cô gái sau 10 năm bỏ nhà đi",
                "Đội tuyển Việt Nam thắng trận giao hữu",
                "Sốc với mức chi tiêu của giới siêu giàu!",
                "Thời tiết ngày mai tại Hà Nội có mưa rào",
            ],
            "sapo": [
                "Cơ quan điều tra vừa công bố các chi tiết mới...",
                "Nghị định có hiệu lực thi hành từ ngày 1/7...",
                "Gặp lại gia đình sau một thập kỷ xa cách...",
                "Các cầu thủ thể hiện phong độ ấn tượng...",
                "Những con số khiến ai cũng phải ngỡ ngàng...",
                "Trung tâm dự báo khí tượng thủy văn thông tin...",
            ],
        })
        self.y_train = np.array([1, 0, 1, 0, 1, 0])

        self.df_test = pd.DataFrame({
            "title": [
                "Tiết lộ lý do nam diễn viên rời khỏi showbiz",
                "Giá vàng SJC tiếp tục ổn định",
            ],
            "sapo": [
                "Sau nhiều đồn đoán, người trong cuộc đã lên tiếng...",
                "Ngân hàng Nhà nước duy trì mức giá bán can thiệp...",
            ],
        })

    def test_version_1_tfidf(self) -> None:
        pipe = ClickbaitFeaturePipeline(variant="v1", word_max_features=50, char_max_features=50)
        X_tr = pipe.fit_transform(self.df_train)
        X_te = pipe.transform(self.df_test)

        self.assertTrue(issparse(X_tr))
        self.assertTrue(issparse(X_te))
        self.assertEqual(X_tr.shape[0], len(self.df_train))
        self.assertEqual(X_te.shape[0], len(self.df_test))
        self.assertEqual(X_tr.shape[1], X_te.shape[1])

        # Test model fit
        clf = LogisticRegression().fit(X_tr, self.y_train)
        preds = clf.predict(X_te)
        self.assertEqual(len(preds), len(self.df_test))

    def test_version_2_stylometric(self) -> None:
        pipe = ClickbaitFeaturePipeline(variant="v2")
        X_tr = pipe.fit_transform(self.df_train)
        X_te = pipe.transform(self.df_test)

        self.assertEqual(X_tr.shape[1], 25)
        self.assertEqual(X_te.shape[1], 25)
        self.assertEqual(X_tr.shape[0], len(self.df_train))
        self.assertEqual(X_te.shape[0], len(self.df_test))

        clf = LogisticRegression().fit(X_tr, self.y_train)
        preds = clf.predict(X_te)
        self.assertEqual(len(preds), len(self.df_test))

    def test_version_3_combined(self) -> None:
        pipe_v1 = ClickbaitFeaturePipeline(variant="v1", word_max_features=50, char_max_features=50)
        X_v1 = pipe_v1.fit_transform(self.df_train)

        pipe_v3 = ClickbaitFeaturePipeline(variant="v3", word_max_features=50, char_max_features=50)
        X_v3 = pipe_v3.fit_transform(self.df_train)
        X_v3_te = pipe_v3.transform(self.df_test)

        self.assertTrue(issparse(X_v3))
        # Dimension of v3 must equal dimension of v1 + 25 stylometric features
        self.assertEqual(X_v3.shape[1], X_v1.shape[1] + 25)
        self.assertEqual(X_v3_te.shape[1], X_v3.shape[1])

        clf = LogisticRegression().fit(X_v3, self.y_train)
        preds = clf.predict(X_v3_te)
        self.assertEqual(len(preds), len(self.df_test))

    def test_invalid_variant(self) -> None:
        with self.assertRaises(ValueError):
            ClickbaitFeaturePipeline(variant="v99")


if __name__ == "__main__":
    unittest.main()
