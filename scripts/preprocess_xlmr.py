"""Offline preprocessing and tokenization for XLM-RoBERTa-large."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ds_project.correction.loader import load_dataset
from ds_project.modeling.xlmr import (
    XLMR_DEFAULT_MODEL_NAME,
    prepare_xlmr_texts,
    save_tokenized_split,
    select_xlmr_max_length,
    summarize_lengths,
    tokenize_split,
    write_json,
)

SPLIT_NAMES = ("train", "dev", "test_seen", "test_unseen")


def run_preprocessing(
    *,
    dataset_path: Path | str,
    model_name: str = XLMR_DEFAULT_MODEL_NAME,
    output_dir: Path | str,
    report_path: Path | str,
    percentile: float = 100.0,
    max_cap: int = 128,
    multiple: int = 8,
    batch_size: int = 1024,
    replace: bool = False,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Tokenize train/dev/test_seen/test_unseen once and write config + length report."""
    out_dir = Path(output_dir)
    rep_path = Path(report_path)
    target_paths = [out_dir / f"{split}.npz" for split in SPLIT_NAMES] + [
        out_dir / "preprocessor_config.json",
        rep_path,
    ]
    if not replace:
        existing = [str(p) for p in target_paths if p.exists()]
        if existing:
            raise FileExistsError(
                f"Output already exists (pass --replace to overwrite): {', '.join(existing)}"
            )

    df = load_dataset(dataset_path)
    splits: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_df = df.loc[df["split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Split '{split_name}' is empty in {dataset_path}")
        splits[split_name] = split_df

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = prepare_xlmr_texts(splits["train"])
    dev_texts = prepare_xlmr_texts(splits["dev"])
    max_length, selection_info = select_xlmr_max_length(
        train_texts,
        dev_texts,
        tokenizer,
        percentile=percentile,
        max_cap=max_cap,
        multiple=multiple,
    )

    split_reports: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        payload, untruncated_lengths = tokenize_split(
            splits[split_name],
            tokenizer,
            max_length=max_length,
            batch_size=batch_size,
        )
        npz_path = out_dir / f"{split_name}.npz"
        save_tokenized_split(
            npz_path,
            payload["record_id"],
            payload["input_ids"],
            payload["attention_mask"],
            payload["lengths"],
        )
        stats = summarize_lengths(untruncated_lengths, max_length=max_length)
        stats["npz_path"] = str(npz_path)
        split_reports[split_name] = stats

    preprocessor_config = {
        "model_name": model_name,
        "max_length": int(max_length),
        "normalization": {
            "unicode_form": "NFC",
            "strip_zero_width": True,
            "collapse_whitespace": True,
            "preserve_case": True,
            "preserve_punctuation": True,
            "word_segmentation": None,
        },
        "max_length_selection": selection_info,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
    }
    config_path = write_json(out_dir / "preprocessor_config.json", preprocessor_config)

    report = {
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "output_dir": str(out_dir),
        "config_path": str(config_path),
        "max_length": int(max_length),
        "max_length_selection": selection_info,
        "splits": split_reports,
    }
    write_json(rep_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset for XLM-R.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/dataset.csv",
        help="Path to processed dataset.csv.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=XLMR_DEFAULT_MODEL_NAME,
        help="HuggingFace tokenizer/model identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/xlmr",
        help="Directory to write tokenized .npz splits and preprocessor_config.json.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "reports/xlmr/xlmr_preprocessing_report.json",
        help="Path to write tokenization length statistics JSON.",
    )
    parser.add_argument("--percentile", type=float, default=100.0)
    parser.add_argument("--max-cap", type=int, default=128)
    parser.add_argument("--multiple", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replace", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_preprocessing(
        dataset_path=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        report_path=args.report_path,
        percentile=args.percentile,
        max_cap=args.max_cap,
        multiple=args.multiple,
        batch_size=args.batch_size,
        replace=args.replace,
    )
    print(f"Preprocessed XLM-R splits with max_length={report['max_length']} -> {report['output_dir']}")
    for split_name in SPLIT_NAMES:
        st = report["splits"][split_name]
        print(
            f"  {split_name}: n={st['count']}, min={st['min']}, median={st['median']:.1f}, "
            f"p95={st['p95']:.1f}, p99={st['p99']:.1f}, max={st['max']}, "
            f"trunc={st['truncation_count']} ({st['truncation_rate']:.4%})"
        )


if __name__ == "__main__":
    main()
