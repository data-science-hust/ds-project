"""
Supports three input variants as defined in project specifications:
- Version 1 (Baseline): TF-IDF Matrix (Word & Character N-grams)
- Version 2 (Extended): 25 Warsaw Stylometric Features (StandardScaler normalized)
- Version 3 (Combined): Concat(TF-IDF Matrix, 25 Stylometric Features)
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from ds_project.features.stylometric import WarsawStylometricExtractor
from ds_project.features.tfidf import TfidfHeadlineVectorizer

InputVariant = Literal["v1", "v2", "v3", "tfidf", "stylometric", "hybrid"]


class ClickbaitFeaturePipeline(BaseEstimator, TransformerMixin):
    """Pipeline extracting and preparing Input Variants for ML models."""

    def __init__(
        self,
        variant: InputVariant = "v1",
        word_ngram_range: tuple[int, int] = (1, 2),
        word_max_features: int = 15000,
        word_min_df: int = 2,
        char_ngram_range: tuple[int, int] = (2, 5),
        char_max_features: int = 20000,
        char_min_df: int = 3,
        use_char_ngrams: bool = True,
        scale_stylometric: bool = True,
    ) -> None:
        canonical_variant = variant.lower()
        if canonical_variant in ("v1", "tfidf"):
            self.variant: InputVariant = "v1"
        elif canonical_variant in ("v2", "stylometric"):
            self.variant = "v2"
        elif canonical_variant in ("v3", "hybrid"):
            self.variant = "v3"
        else:
            raise ValueError(f"Unknown input variant: '{variant}'. Must be 'v1', 'v2', or 'v3'.")

        self.word_ngram_range = word_ngram_range
        self.word_max_features = word_max_features
        self.word_min_df = word_min_df
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.char_min_df = char_min_df
        self.use_char_ngrams = use_char_ngrams
        self.scale_stylometric = scale_stylometric

        self.tfidf_vectorizer_: TfidfHeadlineVectorizer | None = None
        self.stylo_extractor_: WarsawStylometricExtractor | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, X: pd.DataFrame | Sequence[str], y: None = None) -> ClickbaitFeaturePipeline:
        """Fit feature representations strictly on the training set."""
        if self.variant in ("v1", "v3"):
            self.tfidf_vectorizer_ = TfidfHeadlineVectorizer(
                word_ngram_range=self.word_ngram_range,
                word_max_features=self.word_max_features,
                word_min_df=self.word_min_df,
                char_ngram_range=self.char_ngram_range,
                char_max_features=self.char_max_features,
                char_min_df=self.char_min_df,
                use_char_ngrams=self.use_char_ngrams,
            )
            self.tfidf_vectorizer_.fit(X)

        if self.variant in ("v2", "v3"):
            self.stylo_extractor_ = WarsawStylometricExtractor()
            raw_stylo = self.stylo_extractor_.transform(X)
            if self.scale_stylometric:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(raw_stylo)

        return self

    def transform(self, X: pd.DataFrame | Sequence[str]) -> csr_matrix:
        """Transform input into feature matrix X for ML classifiers.

        Returns
        -------
        scipy.sparse.csr_matrix
            Feature matrix:
            - v1: TF-IDF sparse matrix
            - v2: Scaled 25 Warsaw features (as csr_matrix)
            - v3: Horizontal concatenation of TF-IDF and Scaled 25 Warsaw features
        """
        if self.variant == "v1":
            assert self.tfidf_vectorizer_ is not None, "Pipeline must be fitted first"
            return self.tfidf_vectorizer_.transform(X)

        if self.variant == "v2":
            assert self.stylo_extractor_ is not None, "Pipeline must be fitted first"
            stylo = self.stylo_extractor_.transform(X)
            if self.scaler_ is not None:
                stylo = self.scaler_.transform(stylo)
            return csr_matrix(stylo)

        # variant == "v3"
        assert self.tfidf_vectorizer_ is not None and self.stylo_extractor_ is not None
        tfidf_mat = self.tfidf_vectorizer_.transform(X)
        stylo = self.stylo_extractor_.transform(X)
        if self.scaler_ is not None:
            stylo = self.scaler_.transform(stylo)
        stylo_mat = csr_matrix(stylo)
        return hstack([tfidf_mat, stylo_mat], format="csr")

    def get_feature_names_out(self) -> np.ndarray:
        """Return combined feature names for interpretability."""
        if self.variant == "v1":
            assert self.tfidf_vectorizer_ is not None
            return np.array(self.tfidf_vectorizer_.get_feature_names_out())

        if self.variant == "v2":
            assert self.stylo_extractor_ is not None
            return np.array([f"stylo__{f}" for f in self.stylo_extractor_.get_feature_names_out()])

        assert self.tfidf_vectorizer_ is not None and self.stylo_extractor_ is not None
        tfidf_names = self.tfidf_vectorizer_.get_feature_names_out()
        stylo_names = [f"stylo__{f}" for f in self.stylo_extractor_.get_feature_names_out()]
        return np.concatenate([tfidf_names, np.array(stylo_names)])
