from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_project.modeling.sequence import (
    PAD_ID,
    UNK_ID,
    SequencePreprocessor,
    load_sequence_split,
    load_word2vec_format,
    normalize_title,
    segment_vietnamese_text,
    select_max_length,
)

def words(text: str) -> list[str]:
    return normalize_title(text).split()

class SequencePreprocessorTests(unittest.TestCase):
    def test_normalization_preserves_case_and_punctuation(self):
        self.assertEqual(normalize_title("  Ca\u0301CH   hỏi?!…  "), "CáCH hỏi?!…")

    def test_normalization_cleans_invisible_chars_and_preserves_content(self):
        # Cleans zero-width spaces, BOM, and soft hyphens
        dirty = "\u200b\u200b'Cô tiên từ thiện'\ufeff Nguyễn Đỗ Trúc Phương\u00ad..."
        expected = "'Cô tiên từ thiện' Nguyễn Đỗ Trúc Phương..."
        self.assertEqual(normalize_title(dirty), expected)

    def test_fit_uses_only_supplied_training_text_and_is_deterministic(self):
        first = SequencePreprocessor(5).fit(["zeta alpha", "alpha beta"], segmenter=words)
        second = SequencePreprocessor(5).fit(["zeta alpha", "alpha beta"], segmenter=words)
        self.assertEqual(first.token_to_id, second.token_to_id)
        self.assertEqual(first.token_to_id["alpha"], 2)
        self.assertNotIn("dev_only", first.token_to_id)
        self.assertEqual(first.token_to_id["<pad>"], PAD_ID)
        self.assertEqual(first.token_to_id["<unk>"], UNK_ID)

    def test_pre_tokenized_inputs_produce_identical_vocabulary(self):
        raw_texts = ["alpha beta", "beta gamma"]
        tokenized = [t.split() for t in raw_texts]
        p1 = SequencePreprocessor(5).fit(raw_texts, segmenter=words)
        p2 = SequencePreprocessor(5).fit(tokenized, segmenter=words)
        self.assertEqual(p1.token_to_id, p2.token_to_id)

    def test_literal_reserved_tokens_do_not_collide_with_padding_ids(self):
        processor = SequencePreprocessor(2).fit(["<pad> <unk>"], segmenter=words)
        batch = processor.transform(["<pad> <unk>"], segmenter=words)
        self.assertNotEqual(batch.token_ids[0, 0], PAD_ID)
        self.assertNotEqual(batch.token_ids[0, 1], UNK_ID)

    def test_transform_handles_unknowns_padding_and_truncation(self):
        processor = SequencePreprocessor(2).fit(["tin vui"], segmenter=words)
        batch = processor.transform(["tin xa vui", ""], segmenter=words)
        np.testing.assert_array_equal(batch.token_ids, [[2, UNK_ID], [PAD_ID, PAD_ID]])
        np.testing.assert_array_equal(batch.lengths, [2, 0])
        self.assertEqual(processor.truncation_rate(["tin vui extra"], segmenter=words), 1.0)
        self.assertEqual(batch.truncation_rate, 0.5)

    def test_empty_transform_has_two_dimensional_shape(self):
        processor = SequencePreprocessor(3).fit(["headline"], segmenter=words)
        batch = processor.transform([], segmenter=words)
        self.assertEqual(batch.token_ids.shape, (0, 3))
        self.assertEqual(batch.lengths.shape, (0,))
        self.assertEqual(batch.truncation_rate, 0.0)

    def test_select_max_length_uses_train_and_dev_and_caps_value(self):
        self.assertEqual(
            select_max_length(["one two"], ["one two three"], percentile=100, maximum=8, segmenter=words),
            3,
        )
        self.assertEqual(
            select_max_length(["one two three four"], [], percentile=100, maximum=2, segmenter=words),
            2,
        )

    def test_select_max_length_supports_pre_tokenized_lists(self):
        train = [["a", "b"], ["c"]]
        dev = [["d", "e", "f"]]
        self.assertEqual(select_max_length(train, dev, percentile=100, maximum=10), 3)

    def test_select_max_length_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_max_length([], [], segmenter=words)
        with self.assertRaises(ValueError):
            select_max_length(["one"], [], percentile=0, segmenter=words)

    def test_embedding_matrix_reports_oov_and_zeros_padding(self):
        processor = SequencePreprocessor(3).fit(["known missing"], segmenter=words)
        vectors = {"known": np.array([1.0, 2.0], dtype=np.float32)}
        matrix, coverage = processor.build_embedding_matrix(vectors, seed=3)
        self.assertEqual(matrix.shape, (4, 2))
        np.testing.assert_array_equal(matrix[PAD_ID], [0.0, 0.0])
        np.testing.assert_array_equal(matrix[processor.token_to_id["known"]], [1.0, 2.0])
        self.assertEqual((coverage.vocabulary_size, coverage.pretrained_words, coverage.oov_words), (2, 1, 1))
        self.assertEqual(coverage.rate, 0.5)
        self.assertEqual(coverage.coverage_rate, 0.5)
        self.assertEqual(coverage.oov_rate, 0.5)
        self.assertEqual(coverage.oov_tokens, ("missing",))
        cov_dict = coverage.to_dict()
        self.assertEqual(cov_dict["oov_words"], 1)
        self.assertEqual(cov_dict["sample_oov_tokens"], ["missing"])

    def test_embedding_matrix_fallback_lowercase(self):
        processor = SequencePreprocessor(3).fit(["Title Cased"], segmenter=words)
        # Vector store only contains lowercase words
        vectors = {"title": np.array([1.0, 1.0], dtype=np.float32)}
        matrix, coverage = processor.build_embedding_matrix(vectors, fallback_lowercase=True, seed=3)
        self.assertEqual(coverage.pretrained_words, 1)
        self.assertEqual(coverage.oov_words, 1)
        np.testing.assert_array_equal(matrix[processor.token_to_id["Title"]], [1.0, 1.0])

    def test_embedding_matrix_rejects_wrong_dimensions(self):
        processor = SequencePreprocessor(3).fit(["known"], segmenter=words)
        with self.assertRaisesRegex(ValueError, "expected \\(2,\\)"):
            processor.build_embedding_matrix({"known": [1.0]}, vector_size=2)

    def test_save_and_load_preserve_vocabulary_and_transform(self):
        processor = SequencePreprocessor(3, min_frequency=1).fit(["xin chào"], segmenter=words)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessor.json"
            processor.save(path)
            loaded = SequencePreprocessor.load(path)
        self.assertEqual(processor.token_to_id, loaded.token_to_id)
        np.testing.assert_array_equal(
            processor.transform(["xin chào"], segmenter=words).token_ids,
            loaded.transform(["xin chào"], segmenter=words).token_ids,
        )

    def test_unfitted_preprocessor_cannot_transform(self):
        with self.assertRaisesRegex(RuntimeError, "fit"):
            SequencePreprocessor(3).transform(["text"], segmenter=words)

    def test_load_word2vec_format_text_and_gzip(self):
        content = "2 3\nviệt_nam 0.1 0.2 0.3\nhà_nội 0.4 0.5 0.6\n"
        with tempfile.TemporaryDirectory() as directory:
            txt_path = Path(directory) / "w2v.txt"
            txt_path.write_text(content, encoding="utf-8")
            vectors = load_word2vec_format(txt_path)
            self.assertEqual(len(vectors), 2)
            np.testing.assert_allclose(vectors["việt_nam"], [0.1, 0.2, 0.3], rtol=1e-5)

            # Test vocabulary filter
            filtered = load_word2vec_format(txt_path, vocabulary=["hà_nội"])
            self.assertEqual(len(filtered), 1)
            self.assertIn("hà_nội", filtered)

            # Test gzipped format
            gz_path = Path(directory) / "w2v.txt.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                f.write(content)
            gz_vectors = load_word2vec_format(gz_path)
            self.assertEqual(len(gz_vectors), 2)

    def test_load_sequence_split_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            dir_path = Path(directory)
            np.savez_compressed(
                dir_path / "dev.npz",
                record_id=np.array(["rec1", "rec2", "rec3"]),
                token_ids=np.array([[2, 3], [4, 0], [5, 6]]),
                lengths=np.array([2, 1, 2]),
            )

            # Unlabeled load
            ds = load_sequence_split("dev", sequence_dir=dir_path)
            self.assertEqual(len(ds), 3)
            self.assertIsNone(ds.labels)
            self.assertEqual(ds[0]["record_id"], "rec1")

            # Labeled load
            labels_df = pd.DataFrame({
                "record_id": ["rec1", "rec3"],
                "label": [1, 0],
            })
            ds_labeled = load_sequence_split("dev", sequence_dir=dir_path, labels_df=labels_df)
            self.assertEqual(len(ds_labeled), 2)
            self.assertEqual(list(ds_labeled.labels), [1, 0])
            self.assertEqual(ds_labeled[0]["label"], 1)

    def test_segment_vietnamese_text(self):
        text = "Trước Tết, nhà nào cũng vui"
        res = segment_vietnamese_text(text)
        self.assertIsInstance(res, str)
        self.assertIn("Trước_Tết", res)

if __name__ == "__main__":
    unittest.main()

