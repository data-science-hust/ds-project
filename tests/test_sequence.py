from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_project.modeling.sequence import (PAD_ID, UNK_ID, SequencePreprocessor, normalize_title,
                                          select_max_length)


def words(text: str) -> list[str]:
    return normalize_title(text).split()


class SequencePreprocessorTests(unittest.TestCase):
    def test_normalization_preserves_case_and_punctuation(self):
        self.assertEqual(normalize_title("  Ca\u0301CH   hỏi?!…  "), "CáCH hỏi?!…")

    def test_fit_uses_only_supplied_training_text_and_is_deterministic(self):
        first = SequencePreprocessor(5).fit(["zeta alpha", "alpha beta"], segmenter=words)
        second = SequencePreprocessor(5).fit(["zeta alpha", "alpha beta"], segmenter=words)
        self.assertEqual(first.token_to_id, second.token_to_id)
        self.assertEqual(first.token_to_id["alpha"], 2)
        self.assertNotIn("dev_only", first.token_to_id)
        self.assertEqual(first.token_to_id["<pad>"], PAD_ID)
        self.assertEqual(first.token_to_id["<unk>"], UNK_ID)

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

    def test_empty_transform_has_two_dimensional_shape(self):
        processor = SequencePreprocessor(3).fit(["headline"], segmenter=words)
        batch = processor.transform([], segmenter=words)
        self.assertEqual(batch.token_ids.shape, (0, 3))
        self.assertEqual(batch.lengths.shape, (0,))

    def test_select_max_length_uses_train_and_dev_and_caps_value(self):
        self.assertEqual(
            select_max_length(["one two"], ["one two three"], percentile=100, maximum=8, segmenter=words),
            3,
        )
        self.assertEqual(
            select_max_length(["one two three four"], [], percentile=100, maximum=2, segmenter=words),
            2,
        )

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


if __name__ == "__main__":
    unittest.main()
