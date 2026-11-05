from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Protocol

import numpy as np

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
PREPROCESSOR_VERSION = 1
RESERVED_ESCAPE = "\0"
Segmenter = Callable[[str], list[str]]


class VectorStore(Protocol):
    vector_size: int
    def __contains__(self, token: object) -> bool: ...
    def __getitem__(self, token: str) -> object: ...

@dataclass(frozen=True)
class SequenceBatch:
    token_ids: np.ndarray
    lengths: np.ndarray

@dataclass(frozen=True)
class EmbeddingCoverage:
    vocabulary_size: int
    pretrained_words: int
    oov_words: int

    @property
    def rate(self) -> float:
        if self.vocabulary_size == 0:
            return 0.0
        return self.pretrained_words / self.vocabulary_size

def normalize_title(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"title must be a string, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFC", text).split())

def segment_vietnamese(text: str) -> list[str]:
    from pyvi import ViTokenizer

    normalized = normalize_title(text)
    if not normalized:
        return []
    return ViTokenizer.tokenize(normalized).split()

def _vocabulary_key(token: str) -> str:
    if token in (PAD_TOKEN, UNK_TOKEN) or token.startswith(RESERVED_ESCAPE):
        return RESERVED_ESCAPE + token
    return token

def select_max_length(
    train_texts: Iterable[str],
    dev_texts: Iterable[str],
    percentile: float = 99.0,
    maximum: int = 256,
    segmenter: Segmenter = segment_vietnamese,
) -> int:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    lengths = [len(segmenter(text)) for text in chain(train_texts, dev_texts)]
    if not lengths:
        raise ValueError("cannot select a sequence length from empty train and dev data")
    selected = int(np.percentile(lengths, percentile, method="higher"))
    return max(1, min(selected, maximum))

class SequencePreprocessor:
    def __init__(self, max_length: int, min_frequency: int = 1) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        self.max_length = max_length
        self.min_frequency = min_frequency
        self.token_to_id: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.fitted = False

    @property
    def vocabulary_size(self) -> int:
        self._require_fitted()
        return len(self.token_to_id)

    def fit(
        self, train_texts: Iterable[str], segmenter: Segmenter = segment_vietnamese
    ) -> SequencePreprocessor:
        counts: Counter[str] = Counter()
        for text in train_texts:
            counts.update(_vocabulary_key(token) for token in segmenter(text))
        ordered_tokens = sorted(
            (token for token, count in counts.items() if count >= self.min_frequency),
            key=lambda token: (-counts[token], token),
        )
        self.token_to_id = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            **{token: index + 2 for index, token in enumerate(ordered_tokens)},
        }
        self.fitted = True
        return self

    def transform(
        self, texts: Iterable[str], segmenter: Segmenter = segment_vietnamese
    ) -> SequenceBatch:
        self._require_fitted()
        rows: list[list[int]] = []
        lengths: list[int] = []
        for text in texts:
            token_ids = [
                self.token_to_id.get(_vocabulary_key(token), UNK_ID)
                for token in segmenter(text)[: self.max_length]
            ]
            lengths.append(len(token_ids))
            rows.append(token_ids + [PAD_ID] * (self.max_length - len(token_ids)))
        return SequenceBatch(
            token_ids=np.asarray(rows, dtype=np.int64).reshape(-1, self.max_length),
            lengths=np.asarray(lengths, dtype=np.int64),
        )

    def truncation_rate(
        self, texts: Iterable[str], segmenter: Segmenter = segment_vietnamese
    ) -> float:
        self._require_fitted()
        lengths = [len(segmenter(text)) for text in texts]
        if not lengths:
            return 0.0
        return sum(length > self.max_length for length in lengths) / len(lengths)

    def build_embedding_matrix(
        self,
        vectors: VectorStore | Mapping[str, object],
        *,
        vector_size: int | None = None,
        seed: int = 2026,
        random_scale: float = 0.05,
    ) -> tuple[np.ndarray, EmbeddingCoverage]:
        self._require_fitted()
        if random_scale < 0:
            raise ValueError("random_scale must be non-negative")
        inferred_size = vector_size
        if inferred_size is None:
            inferred_size = getattr(vectors, "vector_size", None)
        if inferred_size is None:
            first_vector = next(iter(vectors.values()), None) if isinstance(vectors, Mapping) else None
            if first_vector is None:
                raise ValueError("vector_size is required for an empty or unsupported vector store")
            inferred_size = int(np.asarray(first_vector).size)
        if inferred_size < 1:
            raise ValueError("vector_size must be at least 1")

        matrix = np.random.default_rng(seed).normal(
            0.0, random_scale, size=(self.vocabulary_size, inferred_size)
        ).astype(np.float32)
        matrix[PAD_ID] = 0.0
        covered = 0
        for token, index in self.token_to_id.items():
            vector_token = token[len(RESERVED_ESCAPE):] if token.startswith(RESERVED_ESCAPE) else token
            if index < 2 or vector_token not in vectors:
                continue
            vector = np.asarray(vectors[vector_token], dtype=np.float32)
            if vector.shape != (inferred_size,):
                raise ValueError(
                    f"embedding for {vector_token!r} has shape {vector.shape}; "
                    f"expected ({inferred_size},)"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"embedding for {vector_token!r} contains non-finite values")
            matrix[index] = vector
            covered += 1
        vocabulary_words = self.vocabulary_size - 2
        return matrix, EmbeddingCoverage(
            vocabulary_size=vocabulary_words,
            pretrained_words=covered,
            oov_words=vocabulary_words - covered,
        )

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        payload = {
            "version": PREPROCESSOR_VERSION,
            "max_length": self.max_length,
            "min_frequency": self.min_frequency,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SequencePreprocessor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != PREPROCESSOR_VERSION:
            raise ValueError(f"unsupported preprocessor version: {payload.get('version')!r}")
        preprocessor = cls(
            max_length=int(payload["max_length"]),
            min_frequency=int(payload["min_frequency"]),
        )
        token_to_id = payload["token_to_id"]
        if (
            not isinstance(token_to_id, dict)
            or token_to_id.get(PAD_TOKEN) != PAD_ID
            or token_to_id.get(UNK_TOKEN) != UNK_ID
            or sorted(token_to_id.values()) != list(range(len(token_to_id)))
            or len(set(token_to_id.values())) != len(token_to_id)
        ):
            raise ValueError("invalid or non-contiguous token_to_id vocabulary")
        preprocessor.token_to_id = token_to_id
        preprocessor.fitted = True
        return preprocessor

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit the sequence preprocessor on training text first")
