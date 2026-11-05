from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_project.modeling.sequence import (
    SequencePreprocessor,
    load_word2vec_format,
    segment_vietnamese,
    select_max_length,
)

SPLITS = ("train", "dev", "test_seen", "test_unseen")

def build_features(
    dataset_path: Path,
    output_dir: Path,
    *,
    percentile: float,
    max_length_cap: int,
    min_frequency: int,
    embeddings_path: Path | None = None,
    replace: bool,
) -> dict[str, object]:
    dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    required = {"record_id", "split", "title"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"{dataset_path} is missing required columns: {sorted(missing)}")
    if dataset["record_id"].isna().any() or dataset["record_id"].duplicated().any():
        raise ValueError(f"{dataset_path} must contain unique, non-empty record_id values")
    if dataset["title"].isna().any():
        raise ValueError(f"{dataset_path} contains missing titles")
    if dataset["split"].isna().any():
        raise ValueError(f"{dataset_path} contains records without a split")
    unexpected = set(dataset["split"].dropna()).difference(SPLITS)
    if unexpected:
        raise ValueError(f"{dataset_path} contains unsupported splits: {sorted(unexpected)}")

    frames = {name: dataset.loc[dataset["split"] == name].copy() for name in SPLITS}
    if frames["train"].empty or frames["dev"].empty:
        raise ValueError("train and dev splits must both be non-empty")
    if output_dir.exists() and any(output_dir.iterdir()) and not replace:
        raise FileExistsError(f"{output_dir} is not empty; pass --replace to overwrite outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    segmented_frames: dict[str, list[list[str]]] = {
        name: [segment_vietnamese(t) for t in frames[name]["title"]]
        for name in SPLITS
    }

    max_length = select_max_length(
        segmented_frames["train"],
        segmented_frames["dev"],
        percentile=percentile,
        maximum=max_length_cap,
    )
    preprocessor = SequencePreprocessor(max_length=max_length, min_frequency=min_frequency)
    preprocessor.fit(segmented_frames["train"])
    preprocessor.save(output_dir / "preprocessor.json")

    rel_dataset_path = (
        str(dataset_path.relative_to(ROOT))
        if dataset_path.is_relative_to(ROOT)
        else str(dataset_path)
    )

    report: dict[str, object] = {
        "dataset": rel_dataset_path,
        "tokenizer": "PyVi ViTokenizer",
        "normalization": "Unicode NFC; clean invisible chars; collapse whitespace; preserve case and punctuation",
        "vocabulary_fit_split": "train",
        "max_length_selection_splits": ["train", "dev"],
        "length_percentile": percentile,
        "max_length_cap": max_length_cap,
        "max_length": max_length,
        "min_frequency": min_frequency,
        "vocabulary_size_including_reserved_tokens": preprocessor.vocabulary_size,
        "splits": {},
        "embeddings": None,
    }

    split_report: dict[str, dict[str, object]] = {}
    for split, frame in frames.items():
        batch = preprocessor.transform(segmented_frames[split])
        np.savez_compressed(
            output_dir / f"{split}.npz",
            record_id=frame["record_id"].to_numpy(dtype=str),
            token_ids=batch.token_ids,
            lengths=batch.lengths,
        )
        split_report[split] = {
            "records": len(frame),
            "truncation_rate": batch.truncation_rate,
            "features": f"{split}.npz",
        }
    report["splits"] = split_report

    if embeddings_path is not None:
        vectors = load_word2vec_format(embeddings_path, vocabulary=preprocessor.token_to_id.keys())
        embedding_matrix, coverage = preprocessor.build_embedding_matrix(vectors, fallback_lowercase=True)
        np.save(output_dir / "embedding_matrix.npy", embedding_matrix)
        oov_dict = coverage.to_dict()
        (output_dir / "oov_report.json").write_text(
            json.dumps(oov_dict, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["embeddings"] = {
            "path": str(embeddings_path),
            "matrix_file": "embedding_matrix.npy",
            "vector_dim": int(embedding_matrix.shape[1]),
            "coverage": oov_dict,
        }

    (output_dir / "sequence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sequence features, vocabulary, and embedding matrix for DL models."
    )

    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/sequence")
    parser.add_argument("--embeddings-path", type=Path, default=None, help="Optional path to PhoW2V/word2vec file.")
    parser.add_argument("--length-percentile", type=float, default=99.0)
    parser.add_argument("--max-length-cap", type=int, default=256)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    report = build_features(
        args.dataset,
        args.output_dir,
        percentile=args.length_percentile,
        max_length_cap=args.max_length_cap,
        min_frequency=args.min_frequency,
        embeddings_path=args.embeddings_path,
        replace=args.replace,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
