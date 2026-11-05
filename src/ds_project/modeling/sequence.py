from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Protocol
import numpy as np
import pandas as pd

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
PREPROCESSOR_VERSION = 1
RESERVED_ESCAPE = "\0"
Segmenter = Callable[[str], list[str]]

INVISIBLE_CHARS_RE = re.compile(r"[\u200b-\u200f\ufeff\u00ad]")


class VectorStore(Protocol):
    vector_size: int
    def __contains__(self, token: object) -> bool: ...
    def __getitem__(self, token: str) -> object: ...


@dataclass(frozen=True)
class SequenceBatch:
    token_ids: np.ndarray
    lengths: np.ndarray
    truncation_rate: float = 0.0


@dataclass(frozen=True)
class EmbeddingCoverage:
    vocabulary_size: int
    pretrained_words: int
    oov_words: int
    oov_tokens: tuple[str, ...] = ()

    @property
    def rate(self) -> float:
        if self.vocabulary_size == 0:
            return 0.0
        return self.pretrained_words / self.vocabulary_size

    @property
    def coverage_rate(self) -> float:
        return self.rate

    @property
    def oov_rate(self) -> float:
        if self.vocabulary_size == 0:
            return 0.0
        return self.oov_words / self.vocabulary_size

    def to_dict(self, sample_limit: int = 50) -> dict[str, object]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "pretrained_words": self.pretrained_words,
            "oov_words": self.oov_words,
            "coverage_rate": round(self.coverage_rate, 6),
            "oov_rate": round(self.oov_rate, 6),
            "sample_oov_tokens": list(self.oov_tokens[:sample_limit]),
        }


@dataclass
class SequenceDataset:
    record_id: np.ndarray
    token_ids: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.record_id)

    def __getitem__(self, index: int) -> dict[str, object]:
        item: dict[str, object] = {
            "record_id": str(self.record_id[index]),
            "token_ids": self.token_ids[index],
            "length": int(self.lengths[index]),
        }
        if self.labels is not None:
            item["label"] = int(self.labels[index])
        return item


def normalize_title(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"title must be a string, got {type(text).__name__}")
    cleaned = INVISIBLE_CHARS_RE.sub("", text)
    return " ".join(unicodedata.normalize("NFC", cleaned).split())


def segment_vietnamese(text: str) -> list[str]:
    from pyvi import ViTokenizer

    normalized = normalize_title(text)
    if not normalized:
        return []
    return ViTokenizer.tokenize(normalized).split()


def segment_vietnamese_text(text: str) -> str:
    """Return Vietnamese word-segmented text joined by spaces for PhoBERT tokenization."""
    return " ".join(segment_vietnamese(text))


def _to_token_sequence(item: str | Sequence[str], segmenter: Segmenter) -> list[str]:
    if isinstance(item, str):
        return segmenter(item)
    return list(item)


def _vocabulary_key(token: str) -> str:
    if token in (PAD_TOKEN, UNK_TOKEN) or token.startswith(RESERVED_ESCAPE):
        return RESERVED_ESCAPE + token
    return token


def select_max_length(
    train_texts: Iterable[str | Sequence[str]],
    dev_texts: Iterable[str | Sequence[str]],
    percentile: float = 99.0,
    maximum: int = 256,
    segmenter: Segmenter = segment_vietnamese,
) -> int:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    lengths = [len(_to_token_sequence(text, segmenter)) for text in chain(train_texts, dev_texts)]
    if not lengths:
        raise ValueError("cannot select a sequence length from empty train and dev data")
    selected = int(np.percentile(lengths, percentile, method="higher"))
    return max(1, min(selected, maximum))


def load_word2vec_format(
    path: str | Path,
    vocabulary: Iterable[str] | None = None,
    *,
    encoding: str = "utf-8",
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    vocab_set: set[str] | None = None
    vocab_lower: set[str] | None = None
    if vocabulary is not None:
        vocab_set = {
            t[len(RESERVED_ESCAPE):] if t.startswith(RESERVED_ESCAPE) else t
            for t in vocabulary
        }
        vocab_lower = {t.lower() for t in vocab_set}

    opener = gzip.open if str(path).endswith(".gz") else open
    vectors: dict[str, np.ndarray] = {}

    with opener(path, "rt", encoding=encoding, errors="replace") as f:
        first_line = f.readline()
        if not first_line:
            return vectors

        parts = first_line.strip().split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            pass
        else:
            if len(parts) > 2:
                word = parts[0]
                if vocab_set is None or word in vocab_set or (vocab_lower and word.lower() in vocab_lower):
                    try:
                        vectors[word] = np.asarray(parts[1:], dtype=np.float32)
                    except ValueError:
                        pass

        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(vectors) >= limit:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            if vocab_set is not None:
                if word not in vocab_set and (not vocab_lower or word.lower() not in vocab_lower):
                    continue
            try:
                vectors[word] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue

    return vectors

def load_sequence_split(
    split: str,
    sequence_dir: str | Path = "data/processed/sequence",
    *,
    labels_df: pd.DataFrame | None = None,
    label_col: str = "label",
) -> SequenceDataset:
    path = Path(sequence_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Sequence feature file not found: {path}")

    data = np.load(path)
    record_ids = data["record_id"].astype(str)
    token_ids = data["token_ids"]
    lengths = data["lengths"]

    if labels_df is None:
        return SequenceDataset(record_id=record_ids, token_ids=token_ids, lengths=lengths)

    if "record_id" not in labels_df.columns or label_col not in labels_df.columns:
        raise ValueError(f"labels_df must contain 'record_id' and '{label_col}' columns")

    label_clean = labels_df.dropna(subset=["record_id", label_col]).copy()
    label_map = dict(zip(label_clean["record_id"].astype(str), label_clean[label_col].astype(int)))

    mask = np.array([rid in label_map for rid in record_ids], dtype=bool)
    if not mask.any():
        raise ValueError(f"No records in {split}.npz matched records in labels_df")

    filtered_rids = record_ids[mask]
    filtered_token_ids = token_ids[mask]
    filtered_lengths = lengths[mask]
    labels_array = np.array([label_map[rid] for rid in filtered_rids], dtype=np.int64)

    return SequenceDataset(
        record_id=filtered_rids,
        token_ids=filtered_token_ids,
        lengths=filtered_lengths,
        labels=labels_array,
    )

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
        self,
        train_texts: Iterable[str | Sequence[str]],
        segmenter: Segmenter = segment_vietnamese,
    ) -> SequencePreprocessor:
        counts: Counter[str] = Counter()
        for text in train_texts:
            tokens = _to_token_sequence(text, segmenter)
            counts.update(_vocabulary_key(token) for token in tokens)
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
        self,
        texts: Iterable[str | Sequence[str]],
        segmenter: Segmenter = segment_vietnamese,
    ) -> SequenceBatch:
        self._require_fitted()
        rows: list[list[int]] = []
        lengths: list[int] = []
        num_truncated = 0
        total_count = 0

        for text in texts:
            tokens = _to_token_sequence(text, segmenter)
            total_count += 1
            if len(tokens) > self.max_length:
                num_truncated += 1

            token_ids = [
                self.token_to_id.get(_vocabulary_key(token), UNK_ID)
                for token in tokens[: self.max_length]
            ]
            lengths.append(len(token_ids))
            rows.append(token_ids + [PAD_ID] * (self.max_length - len(token_ids)))

        truncation_rate = (num_truncated / total_count) if total_count > 0 else 0.0
        return SequenceBatch(
            token_ids=np.asarray(rows, dtype=np.int64).reshape(-1, self.max_length),
            lengths=np.asarray(lengths, dtype=np.int64),
            truncation_rate=truncation_rate,
        )

    def truncation_rate(
        self,
        texts: Iterable[str | Sequence[str]],
        segmenter: Segmenter = segment_vietnamese,
    ) -> float:
        self._require_fitted()
        lengths = [len(_to_token_sequence(text, segmenter)) for text in texts]
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
        fallback_lowercase: bool = True,
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
        oov_tokens: list[str] = []

        for token, index in self.token_to_id.items():
            if index < 2:
                continue
            vector_token = token[len(RESERVED_ESCAPE):] if token.startswith(RESERVED_ESCAPE) else token

            lookup_token: str | None = None
            if vector_token in vectors:
                lookup_token = vector_token
            elif fallback_lowercase and vector_token.lower() in vectors:
                lookup_token = vector_token.lower()

            if lookup_token is None:
                oov_tokens.append(vector_token)
                continue

            vector = np.asarray(vectors[lookup_token], dtype=np.float32)
            if vector.shape != (inferred_size,):
                raise ValueError(
                    f"embedding for {lookup_token!r} has shape {vector.shape}; "
                    f"expected ({inferred_size},)"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"embedding for {lookup_token!r} contains non-finite values")
            matrix[index] = vector
            covered += 1

        vocabulary_words = self.vocabulary_size - 2
        return matrix, EmbeddingCoverage(
            vocabulary_size=vocabulary_words,
            pretrained_words=covered,
            oov_words=vocabulary_words - covered,
            oov_tokens=tuple(oov_tokens),
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