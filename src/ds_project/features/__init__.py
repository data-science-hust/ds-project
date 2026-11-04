"""Feature extraction package for clickbait detection."""

from ds_project.features.pipeline import ClickbaitFeaturePipeline
from ds_project.features.stylometric import WarsawStylometricExtractor
from ds_project.features.text_norm import normalize_text
from ds_project.features.tfidf import TfidfHeadlineVectorizer

__all__ = [
    "normalize_text",
    "WarsawStylometricExtractor",
    "TfidfHeadlineVectorizer",
    "ClickbaitFeaturePipeline",
]
