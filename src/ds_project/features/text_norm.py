"""Text normalization utilities for clickbait headline processing.

Preserves casing, punctuation, question marks, and quotation marks
as specified in the project guidelines.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str | None) -> str:
    """Normalize Unicode to NFC and collapse excessive whitespace.

    Parameters
    ----------
    text : str or None
        Raw input string.

    Returns
    -------
    str
        Cleaned text string. Returns empty string if input is None or blank.
    """
    if text is None:
        return ""
    # Normalize Unicode to canonical decomposition / composition (NFC)
    normalized = unicodedata.normalize("NFC", str(text))
    # Replace tabs, non-breaking spaces, and newlines with space
    normalized = re.sub(r"[\s\u00a0\u200b]+", " ", normalized)
    return normalized.strip()
