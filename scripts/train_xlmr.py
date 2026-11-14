"""Hyperparameter tuning and 3-seed fine-tuning for XLM-RoBERTa-large."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ds_project.correction.loader import load_dev, load_train_set
from ds_project.modeling.xlmr import (
    XLMR_DEFAULT_MODEL_NAME,
    XLMR_HIDDEN_SIZE,
    XLMR_SEEDS,
    build_tokenized_dataset,
    load_tokenized_split,
    predict_from_checkpoint,
    save_prediction_csv,
    train_and_evaluate_run,
    write_json,
)

TUNE_LEARNING_RATES = (1e-5, 2e-5, 3e-5)
TUNE_CLASS_WEIGHTS = (None, "balanced")


def _to_rel_posix(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _format_config_id(lr: float, class_weight: str | None) -> str:
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    cw_str = class_weight if class_weight is not None else "none"
    return f"lr{lr_str}_cw_{cw_str}"


def _model_id_for_seed(seed: int) -> str:
    return "xlmrlarge_full_train" if seed == 2026 else f"xlmrlarge_full_train_s{seed}"


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _upsert_csv_row(csv_path: Path, key_col: str, row: dict[str, Any]) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        kept_df = existing_df[existing_df[key_col].astype(str) != str(row[key_col])].copy()
        combined = pd.concat([kept_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    return combined


def _rank_tuning_results(tuning_df: pd.DataFrame) -> pd.Series:
    ranked = tuning_df.copy()
    ranked["_cw_rank"] = ranked["class_weight"].fillna("none").map({"none": 0, "balanced": 1}).fillna(2)
    ranked = ranked.sort_values(
        by=["dev_macro_f1", "dev_accuracy", "dev_macro_f1_050", "lr", "_cw_rank"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def load_training_datasets(
    tokenized_dir: Path,
    *,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[Any, Any]:
    """Load tokenized train/dev splits aligned by record_id with full_corrected & dev_gold."""
    train_df = load_train_set("full_corrected")
    dev_df = load_dev(include_gold=True)
    train_data = load_tokenized_split(tokenized_dir / "train.npz", labels_df=train_df, label_col="label")
    dev_data = load_tokenized_split(tokenized_dir / "dev.npz", labels_df=dev_df, label_col="label")
    train_ds = build_tokenized_dataset(train_data, max_samples=max_train_samples)
    dev_ds = build_tokenized_dataset(dev_data, max_samples=max_eval_samples)
    return train_ds, dev_ds


def run_tuning(
    *,
    tokenized_dir: Path,
    model_name: str,
    reports_dir: Path,
    models_dir: Path,
    train_dataset: Any,
    dev_dataset: Any,
    tokenizer: Any,
    epochs: int = 5,
    batch_size: int = 32,
    grad_accum: int = 1,
    eval_interval: float = 0.5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    precision: str = "auto",
    tune_seed: int = 2026,
    resume: bool = False,
    keep_tune_checkpoints: bool = False,
    num_workers: int = 0,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    """Run the 6-config Dev hyperparameter grid search and write xlmr_best_config.json."""
    tuning_csv = reports_dir / "xlmr_tuning_results.csv"
    completed_ids: set[str] = set()
    if resume and tuning_csv.exists():
        existing_df = pd.read_csv(tuning_csv)
        if "config_id" in existing_df.columns:
            completed_ids = set(existing_df["config_id"].astype(str))

    for lr in TUNE_LEARNING_RATES:
        for cw in TUNE_CLASS_WEIGHTS:
            config_id = _format_config_id(lr, cw)
            if config_id in completed_ids:
                print(f"[tune] Skipping completed config {config_id}")
                continue

            print(f"[tune] Running config {config_id} (lr={lr}, class_weight={cw}, seed={tune_seed})")
            ckpt_dir = models_dir / f"tune_{config_id}"
            log_csv = reports_dir / "logs" / f"tune_{config_id}.csv"
            model_arg = model_factory() if model_factory is not None else model_name

            result = train_and_evaluate_run(
                model_or_name=model_arg,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                output_dir=ckpt_dir,
                tokenizer=tokenizer,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm,
                class_weight=cw,
                eval_interval_epochs=eval_interval,
                precision=precision,
                seed=tune_seed,
                log_csv_path=log_csv,
                num_workers=num_workers,
            )

            default_m = result["default_dev_metrics"]
            tuned_m = result["tuned_dev_metrics"]
            row = {
                "config_id": config_id,
                "model_name": model_name,
                "seed": tune_seed,
                "lr": float(lr),
                "class_weight": cw if cw is not None else "none",
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch_size": result["effective_batch_size"],
                "precision": result["precision"],
                "best_step": result["best_step"],
                "best_epoch": result["best_epoch"],
                "dev_macro_f1_050": float(default_m["macro_f1"]),
                "dev_accuracy_050": float(default_m["accuracy"]),
                "dev_clickbait_f1_050": float(default_m["f1_1"]),
                "optimal_threshold": float(result["optimal_threshold"]),
                "dev_macro_f1": float(tuned_m["macro_f1"]),
                "dev_accuracy": float(tuned_m["accuracy"]),
                "dev_precision_0": float(tuned_m["precision_0"]),
                "dev_recall_0": float(tuned_m["recall_0"]),
                "dev_f1_0": float(tuned_m["f1_0"]),
                "dev_precision_1": float(tuned_m["precision_1"]),
                "dev_recall_1": float(tuned_m["recall_1"]),
                "dev_f1_1": float(tuned_m["f1_1"]),
                "train_seconds": float(result["train_seconds"]),
                "checkpoint_dir": _to_rel_posix(ckpt_dir),
            }
            _upsert_csv_row(tuning_csv, "config_id", row)
            print(
                f"[tune] {config_id}: dev_macro_f1@0.50={row['dev_macro_f1_050']:.4f}, "
                f"tau*={row['optimal_threshold']:.2f}, dev_macro_f1={row['dev_macro_f1']:.4f}, "
                f"time={row['train_seconds']:.1f}s"
            )

            if not keep_tune_checkpoints and ckpt_dir.exists():
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            _cleanup_cuda()

    tuning_df = pd.read_csv(tuning_csv)
    best_row = _rank_tuning_results(tuning_df)
    best_cw_raw = None if pd.isna(best_row["class_weight"]) else str(best_row["class_weight"])
    best_cw = None if best_cw_raw in (None, "", "none") else best_cw_raw

    best_config = {
        "config_id": str(best_row["config_id"]),
        "model_name": model_name,
        "tune_seed": int(best_row["seed"]),
        "lr": float(best_row["lr"]),
        "class_weight": best_cw,
        "epochs": int(best_row["epochs"]),
        "batch_size": int(best_row["batch_size"]),
        "grad_accum": int(best_row["grad_accum"]),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "eval_interval_epochs": float(eval_interval),
        "selection_metric": "tuned_dev_macro_f1",
        "tune_dev_macro_f1": float(best_row["dev_macro_f1"]),
        "tune_dev_macro_f1_050": float(best_row["dev_macro_f1_050"]),
        "tune_optimal_threshold": float(best_row["optimal_threshold"]),
    }
    best_config_path = write_json(reports_dir / "xlmr_best_config.json", best_config)
    print(f"[tune] Best config saved -> {best_config_path}: {best_config['config_id']}")
    return best_config


def run_final(
    *,
    tokenized_dir: Path,
    model_name: str,
    reports_dir: Path,
    models_dir: Path,
    preds_dir: Path,
    train_dataset: Any,
    dev_dataset: Any,
    tokenizer: Any,
    lr: float,
    class_weight: str | None,
    seeds: list[int],
    epochs: int = 5,
    batch_size: int = 32,
    grad_accum: int = 1,
    eval_interval: float = 0.5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    precision: str = "auto",
    resume: bool = False,
    max_eval_samples: int | None = None,
    num_workers: int = 0,
    model_factory: Any | None = None,
) -> pd.DataFrame:
    """Train 3-seed XLM-R runs on train_full_corrected and generate unlabeled test predictions."""
    runs_dev_csv = reports_dir / "xlmr_runs_dev.csv"
    completed_models: set[str] = set()
    if resume and runs_dev_csv.exists():
        existing_df = pd.read_csv(runs_dev_csv)
        if "model_id" in existing_df.columns:
            completed_models = set(existing_df["model_id"].astype(str))

    # Load test splits strictly WITHOUT gold labels
    test_seen_data = load_tokenized_split(tokenized_dir / "test_seen.npz")
    test_unseen_data = load_tokenized_split(tokenized_dir / "test_unseen.npz")
    test_seen_ds = build_tokenized_dataset(test_seen_data, max_samples=max_eval_samples)
    test_unseen_ds = build_tokenized_dataset(test_unseen_data, max_samples=max_eval_samples)
    seen_rids = test_seen_ds.record_ids
    unseen_rids = test_unseen_ds.record_ids

    preproc_cfg_path = tokenized_dir / "preprocessor_config.json"

    for seed in seeds:
        model_id = _model_id_for_seed(seed)
        ckpt_dir = models_dir / model_id
        seen_pred_path = preds_dir / f"{model_id}_test_seen.csv"
        unseen_pred_path = preds_dir / f"{model_id}_test_unseen.csv"

        if (
            resume
            and model_id in completed_models
            and ckpt_dir.exists()
            and seen_pred_path.exists()
            and unseen_pred_path.exists()
        ):
            print(f"[final] Skipping completed seed={seed} ({model_id})")
            continue

        print(f"[final] Training {model_id} (seed={seed}, lr={lr}, class_weight={class_weight})")
        log_csv = reports_dir / "logs" / f"train_log_s{seed}.csv"
        model_arg = model_factory() if model_factory is not None else model_name

        result = train_and_evaluate_run(
            model_or_name=model_arg,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            output_dir=ckpt_dir,
            tokenizer=tokenizer,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
            class_weight=class_weight,
            eval_interval_epochs=eval_interval,
            precision=precision,
            seed=seed,
            log_csv_path=log_csv,
            num_workers=num_workers,
        )

        tau_star = float(result["optimal_threshold"])
        seen_probs = predict_from_checkpoint(
            checkpoint_dir=ckpt_dir,
            dataset=test_seen_ds,
            batch_size=max(batch_size, 64),
            precision=precision,
            num_workers=num_workers,
        )
        unseen_probs = predict_from_checkpoint(
            checkpoint_dir=ckpt_dir,
            dataset=test_unseen_ds,
            batch_size=max(batch_size, 64),
            precision=precision,
            num_workers=num_workers,
        )
        seen_preds = (seen_probs >= tau_star).astype(int)
        unseen_preds = (unseen_probs >= tau_star).astype(int)

        save_prediction_csv(seen_rids, seen_preds, seen_probs, seen_pred_path)
        save_prediction_csv(unseen_rids, unseen_preds, unseen_probs, unseen_pred_path)

        write_json(
            ckpt_dir / "run_config.json",
            {
                "model_id": model_id,
                "model_name": model_name,
                "seed": seed,
                "lr": float(lr),
                "class_weight": class_weight,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch_size": result["effective_batch_size"],
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "eval_interval_epochs": eval_interval,
                "precision": result["precision"],
                "best_step": result["best_step"],
                "best_epoch": result["best_epoch"],
                "optimal_threshold": tau_star,
            },
        )
        if preproc_cfg_path.exists():
            shutil.copy2(preproc_cfg_path, ckpt_dir / "preprocessor_config.json")

        default_m = result["default_dev_metrics"]
        tuned_m = result["tuned_dev_metrics"]
        row = {
            "model_id": model_id,
            "model_family": "XLMR-large",
            "training_cohort": "full_train_20k",
            "correction_level": 1.0,
            "seed": seed,
            "features_dim": XLMR_HIDDEN_SIZE,
            "model_name": model_name,
            "lr": float(lr),
            "class_weight": class_weight if class_weight is not None else "none",
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": result["effective_batch_size"],
            "precision": result["precision"],
            "best_step": result["best_step"],
            "best_epoch": result["best_epoch"],
            "optimal_threshold": tau_star,
            "dev_macro_f1_050": float(default_m["macro_f1"]),
            "dev_accuracy_050": float(default_m["accuracy"]),
            "dev_macro_f1": float(tuned_m["macro_f1"]),
            "dev_accuracy": float(tuned_m["accuracy"]),
            "dev_kappa": float(tuned_m["kappa"]),
            "dev_macro_precision": float(tuned_m["macro_precision"]),
            "dev_macro_recall": float(tuned_m["macro_recall"]),
            "dev_precision_0": float(tuned_m["precision_0"]),
            "dev_recall_0": float(tuned_m["recall_0"]),
            "dev_f1_0": float(tuned_m["f1_0"]),
            "dev_precision_1": float(tuned_m["precision_1"]),
            "dev_recall_1": float(tuned_m["recall_1"]),
            "dev_f1_1": float(tuned_m["f1_1"]),
            "dev_tn": int(tuned_m["tn"]),
            "dev_fp": int(tuned_m["fp"]),
            "dev_fn": int(tuned_m["fn"]),
            "dev_tp": int(tuned_m["tp"]),
            "train_seconds": float(result["train_seconds"]),
            "model_path": _to_rel_posix(ckpt_dir),
            "test_seen_preds": _to_rel_posix(seen_pred_path),
            "test_unseen_preds": _to_rel_posix(unseen_pred_path),
            "log_csv_path": _to_rel_posix(log_csv),
        }
        _upsert_csv_row(runs_dev_csv, "model_id", row)
        print(
            f"[final] {model_id}: dev_macro_f1={row['dev_macro_f1']:.4f} "
            f"(tau*={tau_star:.2f}), preds saved for seen ({len(seen_rids)}) & unseen ({len(unseen_rids)})"
        )
        _cleanup_cuda()

    return pd.read_csv(runs_dev_csv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and train XLM-RoBERTa-large.")
    parser.add_argument("--mode", choices=["tune", "final", "all"], default="all")
    parser.add_argument("--tokenized-dir", type=Path, default=PROJECT_ROOT / "data/processed/xlmr")
    parser.add_argument("--model-name", type=str, default=XLMR_DEFAULT_MODEL_NAME)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports/xlmr")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models/xlmr")
    parser.add_argument("--preds-dir", type=Path, default=PROJECT_ROOT / "predictions")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-interval", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--tune-seed", type=int, default=2026)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in XLMR_SEEDS),
        help="Comma-separated seeds for final training.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-tune-checkpoints", action="store_true")
    parser.add_argument("--lr", type=float, default=None, help="Optional LR override for --mode final.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default=None,
        help="Optional class_weight override for --mode final.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, dev_ds = load_training_datasets(
        args.tokenized_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    best_config: dict[str, Any] | None = None
    if args.mode in ("tune", "all"):
        best_config = run_tuning(
            tokenized_dir=args.tokenized_dir,
            model_name=args.model_name,
            reports_dir=args.reports_dir,
            models_dir=args.models_dir,
            train_dataset=train_ds,
            dev_dataset=dev_ds,
            tokenizer=tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_interval=args.eval_interval,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            precision=args.precision,
            tune_seed=args.tune_seed,
            resume=args.resume,
            keep_tune_checkpoints=args.keep_tune_checkpoints,
            num_workers=args.num_workers,
        )

    if args.mode in ("final", "all"):
        if args.lr is not None:
            final_lr = float(args.lr)
            final_cw = None if args.class_weight in (None, "none") else args.class_weight
        else:
            if best_config is None:
                import json

                best_cfg_path = args.reports_dir / "xlmr_best_config.json"
                if not best_cfg_path.exists():
                    raise FileNotFoundError(
                        f"Best config not found at {best_cfg_path}. Run --mode tune first or pass --lr."
                    )
                best_config = json.loads(best_cfg_path.read_text(encoding="utf-8"))
            final_lr = float(best_config["lr"])
            final_cw = best_config.get("class_weight")
            if args.class_weight is not None:
                final_cw = None if args.class_weight == "none" else args.class_weight

        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        run_final(
            tokenized_dir=args.tokenized_dir,
            model_name=args.model_name,
            reports_dir=args.reports_dir,
            models_dir=args.models_dir,
            preds_dir=args.preds_dir,
            train_dataset=train_ds,
            dev_dataset=dev_ds,
            tokenizer=tokenizer,
            lr=final_lr,
            class_weight=final_cw,
            seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_interval=args.eval_interval,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            precision=args.precision,
            resume=args.resume,
            max_eval_samples=args.max_eval_samples,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
