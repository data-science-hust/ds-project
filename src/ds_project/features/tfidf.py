"""TF-IDF feature extraction pipeline for machine learning models.

Builds word n-grams and character n-grams representation, fitted strictly on train.
Preserves punctuation and casing cues relevant for clickbait headlines.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from ds_project.features.text_norm import normalize_text


class TfidfHeadlineVectorizer(BaseEstimator, TransformerMixin):
    """Combined Word and Character n-gram TF-IDF vectorizer for headlines."""

    def __init__(
        self,
        word_ngram_range: tuple[int, int] = (1, 2),
        word_max_features: int = 15000,
        word_min_df: int = 2,
        char_ngram_range: tuple[int, int] = (2, 5),
        char_max_features: int = 20000,
        char_min_df: int = 3,
        use_char_ngrams: bool = True,
        sublinear_tf: bool = True,
    ) -> None:
        self.word_ngram_range = word_ngram_range
        self.word_max_features = word_max_features
        self.word_min_df = word_min_df
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.char_min_df = char_min_df
        self.use_char_ngrams = use_char_ngrams
        self.sublinear_tf = sublinear_tf

        self._build_union()

    def _build_union(self, n_docs: int = 100) -> None:
        effective_word_min_df = self.word_min_df if n_docs >= 20 else 1
        effective_char_min_df = self.char_min_df if n_docs >= 20 else 1

        # Word vectorizer: tokenizes words, keeps casing/punctuation
        word_vec = TfidfVectorizer(
            preprocessor=normalize_text,
            ngram_range=self.word_ngram_range,
            max_features=self.word_max_features,
            min_df=effective_word_min_df,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b|[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~…“”‘’]",
        )

        if not self.use_char_ngrams:
            self.vectorizer_ = word_vec
        else:
            char_vec = TfidfVectorizer(
                preprocessor=normalize_text,
                analyzer="char_wb",
                ngram_range=self.char_ngram_range,
                max_features=self.char_max_features,
                min_df=effective_char_min_df,
                sublinear_tf=self.sublinear_tf,
            )
            self.vectorizer_ = FeatureUnion([
                ("word", word_vec),
                ("char", char_vec),
            ])

    def _extract_text_series(self, X: pd.DataFrame | Sequence[str]) -> list[str]:
        if isinstance(X, pd.DataFrame):
            if "title" in X.columns:
                return X["title"].fillna("").astype(str).tolist()
            return X.iloc[:, 0].fillna("").astype(str).tolist()
        return [str(x) if x is not None else "" for x in X]

    def fit(self, X: pd.DataFrame | Sequence[str], y: None = None) -> TfidfHeadlineVectorizer:
        texts = self._extract_text_series(X)
        self._build_union(n_docs=len(texts))
        self.vectorizer_.fit(texts)
        return self

    def transform(self, X: pd.DataFrame | Sequence[str]) -> csr_matrix:
        texts = self._extract_text_series(X)
        res = self.vectorizer_.transform(texts)
        if not isinstance(res, csr_matrix):
            res = csr_matrix(res)
        return res

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return self.vectorizer_.get_feature_names_out()
