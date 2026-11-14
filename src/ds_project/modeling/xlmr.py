"""Preprocessing, pure-PyTorch fine-tuning, thresholding, and manifest utilities for XLM-R."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ds_project.evaluation.eval1 import calculate_metrics
from ds_project.modeling.sequence import normalize_title

XLMR_DEFAULT_MODEL_NAME = "xlm-roberta-large"
XLMR_HIDDEN_SIZE = 1024
XLMR_SEEDS = (2026, 42, 123)

MODEL_MANIFEST_COLUMNS = (
    "model_id",
    "model_family",
    "training_cohort",
    "correction_level",
    "seed",
    "features_dim",
    "optimal_threshold",
    "dev_macro_f1",
    "dev_accuracy",
    "dev_clickbait_f1",
    "test_seen_macro_f1",
    "test_seen_accuracy",
    "test_unseen_macro_f1",
    "test_unseen_accuracy",
    "model_path",
    "test_seen_preds",
    "test_unseen_preds",
)


def prepare_xlmr_texts(df_or_texts: pd.DataFrame | Iterable[str]) -> list[str]:
    """Apply Unicode NFC and whitespace normalization while preserving case and punctuation.

    Does not apply Vietnamese word segmentation (PyVi), per §6 of the project plan.
    """
    if isinstance(df_or_texts, pd.DataFrame):
        if "title" not in df_or_texts.columns:
            raise ValueError("DataFrame must contain a 'title' column")
        if df_or_texts["title"].isna().any():
            raise ValueError("DataFrame contains missing 'title' values")
        raw_texts: Iterable[str] = df_or_texts["title"].tolist()
    else:
        raw_texts = df_or_texts

    return [normalize_title(str(text)) for text in raw_texts]


def round_up_to_multiple(value: int, multiple: int = 8) -> int:
    """Round a positive integer up to the nearest multiple."""
    if value < 1:
        raise ValueError("value must be at least 1")
    if multiple < 1:
        raise ValueError("multiple must be at least 1")
    return ((value + multiple - 1) // multiple) * multiple


def compute_token_lengths(
    texts: Sequence[str],
    tokenizer: Any,
    batch_size: int = 1024,
) -> list[int]:
    """Compute subword token lengths (including special tokens) without truncation."""
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def summarize_lengths(lengths: Sequence[int], max_length: int) -> dict[str, float | int]:
    """Compute summary statistics and truncation rate for a split's token lengths."""
    if not lengths:
        raise ValueError("Cannot summarize empty token lengths")
    arr = np.asarray(lengths, dtype=np.int64)
    trunc_count = int((arr > max_length).sum())
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "truncation_count": trunc_count,
        "truncation_rate": float(trunc_count / arr.size),
    }


def select_xlmr_max_length(
    train_texts: Sequence[str],
    dev_texts: Sequence[str],
    tokenizer: Any,
    *,
    percentile: float = 100.0,
    max_cap: int = 128,
    multiple: int = 8,
) -> tuple[int, dict[str, Any]]:
    """Select max_length from train + dev subword lengths (never inspecting test splits)."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if max_cap < 1:
        raise ValueError("max_cap must be at least 1")
    if not train_texts or not dev_texts:
        raise ValueError("train_texts and dev_texts must both be non-empty")

    train_lengths = compute_token_lengths(train_texts, tokenizer)
    dev_lengths = compute_token_lengths(dev_texts, tokenizer)
    combined = train_lengths + dev_lengths

    raw_selected = int(np.percentile(combined, percentile, method="higher"))
    rounded = round_up_to_multiple(max(1, raw_selected), multiple=multiple)
    selected = max(1, min(rounded, max_cap))

    info: dict[str, Any] = {
        "percentile": float(percentile),
        "max_cap": int(max_cap),
        "multiple": int(multiple),
        "raw_selected_length": raw_selected,
        "rounded_length": rounded,
        "max_length": selected,
        "train_dev_combined": summarize_lengths(combined, selected),
    }
    return selected, info


def tokenize_split(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    batch_size: int = 1024,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Tokenize a split DataFrame with fixed max_length and record untruncated lengths."""
    if "record_id" not in df.columns:
        raise ValueError("DataFrame must contain 'record_id'")
    if df["record_id"].isna().any() or df["record_id"].duplicated().any():
        raise ValueError("DataFrame must contain unique, non-empty 'record_id' values")

    texts = prepare_xlmr_texts(df)
    untruncated_lengths = compute_token_lengths(texts, tokenizer, batch_size=batch_size)

    input_id_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_id_chunks.append(encoded["input_ids"].astype(np.int32))
        mask_chunks.append(encoded["attention_mask"].astype(np.uint8))

    if input_id_chunks:
        input_ids = np.concatenate(input_id_chunks, axis=0)
        attention_mask = np.concatenate(mask_chunks, axis=0)
    else:
        input_ids = np.zeros((0, max_length), dtype=np.int32)
        attention_mask = np.zeros((0, max_length), dtype=np.uint8)

    effective_lengths = attention_mask.sum(axis=1).astype(np.int32)
    payload = {
        "record_id": df["record_id"].astype(str).to_numpy(),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": effective_lengths,
    }
    return payload, untruncated_lengths


def save_tokenized_split(
    out_path: str | Path,
    record_ids: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    lengths: np.ndarray,
) -> Path:
    """Save tokenized split arrays to compressed .npz format."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        record_id=np.asarray(record_ids, dtype=str),
        input_ids=np.asarray(input_ids, dtype=np.int32),
        attention_mask=np.asarray(attention_mask, dtype=np.uint8),
        lengths=np.asarray(lengths, dtype=np.int32),
    )
    return path


def load_tokenized_split(
    npz_path: str | Path,
    *,
    labels_df: pd.DataFrame | None = None,
    label_col: str = "label",
) -> dict[str, np.ndarray]:
    """Load a tokenized .npz split and optionally align with a labels DataFrame by record_id."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenized split file not found: {path}")

    with np.load(path) as data:
        record_ids = data["record_id"].astype(str)
        input_ids = data["input_ids"].astype(np.int64)
        attention_mask = data["attention_mask"].astype(np.int64)
        lengths = data["lengths"].astype(np.int64)

    if labels_df is None:
        return {
            "record_id": record_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": lengths,
        }

    if "record_id" not in labels_df.columns or label_col not in labels_df.columns:
        raise ValueError(f"labels_df must contain 'record_id' and '{label_col}' columns")

    clean_df = labels_df.dropna(subset=["record_id", label_col]).copy()
    label_map = dict(zip(clean_df["record_id"].astype(str), clean_df[label_col].astype(int)))

    mask = np.array([rid in label_map for rid in record_ids], dtype=bool)
    if not mask.any():
        raise ValueError(f"No records in {path.name} matched records in labels_df")

    filtered_rids = record_ids[mask]
    labels = np.array([label_map[rid] for rid in filtered_rids], dtype=np.int64)
    return {
        "record_id": filtered_rids,
        "input_ids": input_ids[mask],
        "attention_mask": attention_mask[mask],
        "lengths": lengths[mask],
        "labels": labels,
    }


def compute_xlmr_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float | int]:
    """Compute binary classification metrics reusing ds_project.evaluation.eval1.calculate_metrics."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    metrics = calculate_metrics(yt, yp)
    metrics["macro_precision"] = 0.5 * (float(metrics["precision_0"]) + float(metrics["precision_1"]))
    metrics["macro_recall"] = 0.5 * (float(metrics["recall_0"]) + float(metrics["recall_1"]))
    return metrics


def find_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    *,
    start: float = 0.25,
    stop: float = 0.75,
    step: float = 0.01,
) -> tuple[float, dict[str, float | int]]:
    """Scan probability threshold on Dev to maximize macro_f1 (tie-break closest to 0.50)."""
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")

    yt = np.asarray(y_true, dtype=int)
    pr = np.asarray(probs, dtype=float)
    if yt.shape[0] == 0 or yt.shape[0] != pr.shape[0]:
        raise ValueError("y_true and probs must be non-empty and have matching lengths")

    n_steps = int(round((stop - start) / step)) + 1
    candidates = [round(start + i * step, 4) for i in range(n_steps)]
    candidates = [th for th in candidates if th <= stop + 1e-9]
    if not candidates:
        candidates = [0.5]

    best_th = 0.5
    best_f1 = -1.0
    best_metrics: dict[str, float | int] = {}

    for th in candidates:
        preds = (pr >= th).astype(int)
        metrics = compute_xlmr_metrics(yt, preds)
        f1_val = float(metrics["macro_f1"])
        is_better = f1_val > best_f1 + 1e-12
        is_tie_closer = abs(f1_val - best_f1) <= 1e-12 and abs(th - 0.50) < abs(best_th - 0.50)
        if is_better or is_tie_closer:
            best_f1 = f1_val
            best_th = th
            best_metrics = metrics

    return float(best_th), best_metrics


def set_all_seeds(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    """Compute balanced binary class weights: n_samples / (2 * count_c)."""
    arr = np.asarray(labels, dtype=int)
    if arr.size == 0:
        raise ValueError("labels must not be empty")
    counts = np.bincount(arr, minlength=2)
    if counts.shape[0] > 2 or counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Expected non-zero counts for binary classes 0 and 1, got {counts.tolist()}")
    total = float(arr.size)
    return np.array([total / (2.0 * counts[0]), total / (2.0 * counts[1])], dtype=np.float32)


def resolve_amp_dtype(precision: str, device: Any) -> Any:
    """Resolve PyTorch AMP dtype from precision flag and target device."""
    import torch

    normalized = precision.lower().strip()
    if normalized not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision '{precision}'; expected auto, bf16, fp16, or fp32")

    if getattr(device, "type", "cpu") != "cuda" or normalized == "fp32":
        return None
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    # auto: prefer bf16 on Ampere/Ada/Hopper (e.g., RTX 4090) for numerical stability
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_tokenized_dataset(
    split_data: dict[str, np.ndarray],
    *,
    max_samples: int | None = None,
) -> Any:
    """Construct a PyTorch Dataset from a loaded tokenized split dictionary."""
    import torch

    class _TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, data: dict[str, np.ndarray]) -> None:
            limit = len(data["record_id"]) if max_samples is None else min(max_samples, len(data["record_id"]))
            self.record_ids = data["record_id"][:limit]
            self.input_ids = torch.as_tensor(data["input_ids"][:limit], dtype=torch.long)
            self.attention_mask = torch.as_tensor(data["attention_mask"][:limit], dtype=torch.long)
            self.lengths = data["lengths"][:limit].astype(np.int64)
            has_labels = "labels" in data and data["labels"] is not None
            self.labels = torch.as_tensor(data["labels"][:limit], dtype=torch.long) if has_labels else None

        def __len__(self) -> int:
            return len(self.record_ids)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item: dict[str, Any] = {
                "record_id": str(self.record_ids[idx]),
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "length": int(self.lengths[idx]),
            }
            if self.labels is not None:
                item["labels"] = self.labels[idx]
            return item

    return _TokenizedDataset(split_data)


def dynamic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Trim padded batch tensors to the longest actual sequence length in the batch."""
    import torch

    max_len = max(1, max(int(item["length"]) for item in batch))
    input_ids = torch.stack([item["input_ids"][:max_len] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"][:max_len] for item in batch], dim=0)
    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([item["labels"] for item in batch], dim=0)
    return out


def build_optimizer_grouped_parameters(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Disable weight decay for bias and LayerNorm weights following standard transformer practice."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def predict_probabilities(
    model: Any,
    data_loader: Any,
    device: Any,
    *,
    amp_dtype: Any = None,
    criterion: Any = None,
) -> tuple[np.ndarray, float | None]:
    """Run inference over a DataLoader and return class-1 probabilities and optional mean loss."""
    import torch

    model.eval()
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0

    with torch.inference_mode():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None),
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if criterion is not None and labels is not None:
                    loss = criterion(logits.float(), labels)
                    batch_n = int(labels.size(0))
                    total_loss += float(loss.item()) * batch_n
                    total_items += batch_n

            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    probs_concat = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0,), dtype=np.float32)
    mean_loss = (total_loss / total_items) if total_items > 0 else None
    return probs_concat, mean_loss


def train_and_evaluate_run(
    *,
    model_or_name: str | Path | Any,
    train_dataset: Any,
    dev_dataset: Any,
    output_dir: str | Path,
    tokenizer: Any = None,
    lr: float = 2e-5,
    epochs: int = 5,
    batch_size: int = 32,
    grad_accum: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    class_weight: str | None = None,
    eval_interval_epochs: float = 0.5,
    precision: str = "auto",
    seed: int = 2026,
    threshold_start: float = 0.25,
    threshold_stop: float = 0.75,
    threshold_step: float = 0.01,
    log_csv_path: str | Path | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Fine-tune XLM-R with pure PyTorch, checkpoint best Dev Macro-F1 (0.50), then scan tau*."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1 or grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be >= 1")
    if class_weight not in (None, "balanced"):
        raise ValueError(f"Unsupported class_weight '{class_weight}'; expected None or 'balanced'")

    set_all_seeds(seed)
    t_start = time.time()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(precision, device)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if isinstance(model_or_name, (str, Path)):
        model = AutoModelForSequenceClassification.from_pretrained(str(model_or_name), num_labels=2)
    else:
        model = model_or_name
    model.to(device)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=dynamic_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=max(batch_size, 32),
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_labels = np.array([int(train_dataset[i]["labels"]) for i in range(len(train_dataset))], dtype=np.int64)
    dev_labels = np.array([int(dev_dataset[i]["labels"]) for i in range(len(dev_dataset))], dtype=np.int64)

    if class_weight == "balanced":
        weights_tensor = torch.tensor(compute_class_weights(train_labels), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    grouped_params = build_optimizer_grouped_parameters(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(grouped_params, lr=lr)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(round(warmup_ratio * total_updates))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    eval_every_updates = (
        max(1, int(round(updates_per_epoch * eval_interval_epochs)))
        if eval_interval_epochs > 0
        else updates_per_epoch
    )

    best_dev_macro_f1_050 = -1.0
    best_step = 0
    best_epoch = 0.0
    best_default_metrics: dict[str, float | int] = {}
    history: list[dict[str, Any]] = []

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    def _evaluate_and_maybe_checkpoint(cur_epoch: float, running_train_loss: float) -> None:
        nonlocal best_dev_macro_f1_050, best_step, best_epoch, best_default_metrics
        dev_probs, dev_loss = predict_probabilities(
            model,
            dev_loader,
            device,
            amp_dtype=amp_dtype,
            criterion=criterion,
        )
        dev_preds_050 = (dev_probs >= 0.50).astype(int)
        metrics_050 = compute_xlmr_metrics(dev_labels, dev_preds_050)
        is_best = float(metrics_050["macro_f1"]) > best_dev_macro_f1_050 + 1e-12
        if is_best:
            best_dev_macro_f1_050 = float(metrics_050["macro_f1"])
            best_step = global_step
            best_epoch = cur_epoch
            best_default_metrics = dict(metrics_050)
            model.save_pretrained(out_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(out_dir)

        history.append({
            "step": global_step,
            "epoch": round(cur_epoch, 4),
            "train_loss": round(running_train_loss, 6),
            "dev_loss": round(float(dev_loss), 6) if dev_loss is not None else np.nan,
            "dev_macro_f1_050": round(float(metrics_050["macro_f1"]), 6),
            "dev_accuracy_050": round(float(metrics_050["accuracy"]), 6),
            "dev_clickbait_f1_050": round(float(metrics_050["f1_1"]), 6),
            "is_best": bool(is_best),
        })

    for epoch_idx in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        updates_this_epoch = 0
        last_eval_step = global_step

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None),
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                raw_loss = criterion(outputs.logits.float(), labels)
                scaled_loss = raw_loss / grad_accum

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            bs = int(labels.size(0))
            epoch_loss_sum += float(raw_loss.item()) * bs
            epoch_sample_count += bs

            is_accum_boundary = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader)
            if is_accum_boundary:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                updates_this_epoch += 1

                is_epoch_end = (batch_idx + 1) == len(train_loader)
                hit_interval = (global_step - last_eval_step) >= eval_every_updates
                if hit_interval or is_epoch_end:
                    cur_epoch = epoch_idx + (updates_this_epoch / updates_per_epoch)
                    avg_loss = epoch_loss_sum / max(1, epoch_sample_count)
                    _evaluate_and_maybe_checkpoint(cur_epoch, avg_loss)
                    last_eval_step = global_step
                    model.train()

    if log_csv_path is not None:
        log_path = Path(log_csv_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(log_path, index=False)

    # Reload best checkpoint selected at default 0.50, then scan tau* once on Dev
    best_model = type(model).from_pretrained(out_dir)
    best_model.to(device)
    best_dev_probs, _ = predict_probabilities(best_model, dev_loader, device, amp_dtype=amp_dtype)
    tau_star, tuned_metrics = find_optimal_threshold(
        dev_labels,
        best_dev_probs,
        start=threshold_start,
        stop=threshold_stop,
        step=threshold_step,
    )

    train_seconds = time.time() - t_start
    return {
        "seed": seed,
        "lr": lr,
        "class_weight": class_weight,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "precision": str(amp_dtype) if amp_dtype is not None else "fp32",
        "best_step": best_step,
        "best_epoch": round(best_epoch, 4),
        "default_dev_metrics": best_default_metrics,
        "optimal_threshold": tau_star,
        "tuned_dev_metrics": tuned_metrics,
        "dev_probs": best_dev_probs,
        "history": history,
        "train_seconds": round(train_seconds, 2),
        "checkpoint_dir": str(out_dir),
    }


def predict_from_checkpoint(
    *,
    checkpoint_dir: str | Path,
    dataset: Any,
    batch_size: int = 64,
    precision: str = "auto",
    num_workers: int = 0,
) -> np.ndarray:
    """Load a saved XLM-R checkpoint and return class-1 probabilities over dataset."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(precision, device)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    probs, _ = predict_probabilities(model, loader, device, amp_dtype=amp_dtype)
    return probs


def save_prediction_csv(
    record_ids: Sequence[str] | np.ndarray,
    preds: Sequence[int] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    out_path: str | Path,
) -> Path:
    """Save prediction CSV matching predictions/* schema (record_id,predicted_label,confidence_score)."""
    rids = np.asarray(record_ids, dtype=str)
    pred_arr = np.asarray(preds, dtype=int)
    prob_arr = np.asarray(probs, dtype=float)

    if not (len(rids) == len(pred_arr) == len(prob_arr)):
        raise ValueError("record_ids, preds, and probs must have identical lengths")
    if len(rids) == 0:
        raise ValueError("Cannot save empty prediction file")
    if np.isnan(prob_arr).any():
        raise ValueError("probs contains NaN values")
    if not set(np.unique(pred_arr)).issubset({0, 1}):
        raise ValueError("preds must contain only binary values {0, 1}")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "record_id": rids,
        "predicted_label": pred_arr,
        "confidence_score": prob_arr,
    }).to_csv(path, index=False)
    return path


def upsert_model_manifest(
    manifest_path: str | Path,
    new_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Upsert rows into data/manifests/model_manifest.csv keyed by model_id, preserving existing rows."""
    if not new_rows:
        raise ValueError("new_rows must not be empty")

    for row in new_rows:
        missing_cols = [col for col in MODEL_MANIFEST_COLUMNS if col not in row]
        if missing_cols:
            raise ValueError(f"Manifest row is missing required columns: {missing_cols}")

    path = Path(manifest_path)
    incoming_ids = {str(row["model_id"]) for row in new_rows}
    if len(incoming_ids) != len(new_rows):
        raise ValueError("new_rows contains duplicate model_id entries")

    if path.exists():
        existing_df = pd.read_csv(path)
        if list(existing_df.columns) != list(MODEL_MANIFEST_COLUMNS):
            raise ValueError(
                f"Existing manifest columns {list(existing_df.columns)} do not match "
                f"expected schema {list(MODEL_MANIFEST_COLUMNS)}"
            )
        kept_df = existing_df[~existing_df["model_id"].astype(str).isin(incoming_ids)].copy()
    else:
        kept_df = pd.DataFrame(columns=list(MODEL_MANIFEST_COLUMNS))

    new_df = pd.DataFrame(new_rows, columns=list(MODEL_MANIFEST_COLUMNS))
    combined = pd.concat([kept_df, new_df], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write formatted UTF-8 JSON artifact."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
