"""Offline unit and smoke tests for XLM-R preprocessing, training, evaluation, and manifest logic."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_xlmr
import preprocess_xlmr
import train_xlmr
from ds_project.modeling.xlmr import (
    XLMR_HIDDEN_SIZE,
    build_tokenized_dataset,
    compute_xlmr_metrics,
    find_optimal_threshold,
    load_tokenized_split,
    predict_from_checkpoint,
    prepare_xlmr_texts,
    save_prediction_csv,
    save_tokenized_split,
    select_xlmr_max_length,
    tokenize_split,
    train_and_evaluate_run,
    upsert_model_manifest,
)


class _DummyTokenizer:
    """Deterministic offline tokenizer mimicking HuggingFace BatchEncoding API."""

    vocab_size = 128

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        max_length=None,
        return_attention_mask=True,
        return_tensors=None,
    ):
        encoded_ids: list[list[int]] = []
        encoded_masks: list[list[int]] = []
        for text in texts:
            words = [w for w in str(text).split(" ") if w]
            tokens = [10 + (sum(ord(ch) for ch in w) % 100) for w in words]
            if add_special_tokens:
                tokens = [0] + tokens + [2]
            if truncation and max_length is not None:
                tokens = tokens[:max_length]
            mask = [1] * len(tokens)
            if padding == "max_length" and max_length is not None:
                pad_len = max(0, max_length - len(tokens))
                tokens = tokens + [1] * pad_len
                mask = mask + [0] * pad_len
            encoded_ids.append(tokens)
            encoded_masks.append(mask)

        out: dict[str, object] = {}
        if return_tensors == "np":
            out["input_ids"] = np.asarray(encoded_ids, dtype=np.int64)
            if return_attention_mask:
                out["attention_mask"] = np.asarray(encoded_masks, dtype=np.int64)
        else:
            out["input_ids"] = encoded_ids
            if return_attention_mask:
                out["attention_mask"] = encoded_masks
        return out


def _build_tiny_xlmr_model():
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    cfg = XLMRobertaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=2,
    )
    return XLMRobertaForSequenceClassification(cfg)


class TestXlmrPipeline(unittest.TestCase):
    def test_prepare_xlmr_texts_preserves_case_and_punctuation(self) -> None:
        raw_titles = [
            "SỐC!!   Tin\u200b mới:   Giá xăng tăng?",
            "Ti\u00ea\u0301ng Vie\u0323\u0302t chuẩn NFC — không tách từ!",
        ]
        df = pd.DataFrame({"title": raw_titles})
        cleaned = prepare_xlmr_texts(df)

        self.assertEqual(cleaned[0], "SỐC!! Tin mới: Giá xăng tăng?")
        self.assertNotIn("\u200b", cleaned[0])
        self.assertIn("SỐC!!", cleaned[0])
        self.assertIn("?", cleaned[0])
        self.assertNotIn("_", cleaned[1])
        self.assertIn("Tiếng", cleaned[1])

    def test_select_xlmr_max_length_and_tokenized_npz_roundtrip(self) -> None:
        tok = _DummyTokenizer()
        train_texts = ["a b c", "d e f g h"]  # lengths 5, 7
        dev_texts = ["a b c d e f g h i j"]  # length 12
        max_len, info = select_xlmr_max_length(
            train_texts,
            dev_texts,
            tok,
            percentile=100.0,
            max_cap=128,
            multiple=8,
        )
        self.assertEqual(info["raw_selected_length"], 12)
        self.assertEqual(max_len, 16)

        df = pd.DataFrame({
            "record_id": ["r1", "r2", "r3"],
            "title": ["Tiêu đề 1", "Tiêu đề số 2", "Tiêu đề thứ 3"],
        })
        payload, _ = tokenize_split(df, tok, max_length=max_len, batch_size=2)
        self.assertEqual(payload["input_ids"].shape, (3, 16))
        self.assertEqual(payload["attention_mask"].shape, (3, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "dev.npz"
            save_tokenized_split(
                npz_path,
                payload["record_id"],
                payload["input_ids"],
                payload["attention_mask"],
                payload["lengths"],
            )
            labels_df = pd.DataFrame({
                "record_id": ["r3", "r1"],
                "label": [1, 0],
            })
            aligned = load_tokenized_split(npz_path, labels_df=labels_df, label_col="label")
            self.assertEqual(list(aligned["record_id"]), ["r1", "r3"])
            self.assertEqual(list(aligned["labels"]), [0, 1])

    def test_find_optimal_threshold_improves_macro_f1_and_ties(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs = np.array([0.10, 0.20, 0.28, 0.30, 0.36, 0.42, 0.45, 0.80])
        m_050 = compute_xlmr_metrics(y_true, (probs >= 0.50).astype(int))
        tau, best_m = find_optimal_threshold(y_true, probs, start=0.25, stop=0.75, step=0.01)
        self.assertGreater(best_m["macro_f1"], m_050["macro_f1"])
        self.assertGreaterEqual(tau, 0.31)
        self.assertLessEqual(tau, 0.36)

        # Tie-breaking closest to 0.50
        y_tie = np.array([0, 1])
        p_tie = np.array([0.20, 0.80])
        tau_tie, tie_m = find_optimal_threshold(y_tie, p_tie, start=0.25, stop=0.75, step=0.05)
        self.assertEqual(tie_m["macro_f1"], 1.0)
        self.assertAlmostEqual(tau_tie, 0.50, places=4)

    def test_save_prediction_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "preds.csv"
            save_prediction_csv(["id1", "id2"], [0, 1], [0.12, 0.88], out_csv)
            loaded = pd.read_csv(out_csv)
            self.assertEqual(list(loaded.columns), ["record_id", "predicted_label", "confidence_score"])
            self.assertEqual(len(loaded), 2)

            with self.assertRaises(ValueError):
                save_prediction_csv(["id1"], [0], [np.nan], out_csv)

    def test_upsert_model_manifest_preserves_existing_ml_rows(self) -> None:
        real_manifest = PROJECT_ROOT / "data/manifests/model_manifest.csv"
        real_df = pd.read_csv(real_manifest)
        incoming_ids = {
            "xlmrlarge_full_train",
            "xlmrlarge_full_train_s42",
            "xlmrlarge_full_train_s123",
        }
        non_incoming_df = real_df[~real_df["model_id"].astype(str).isin(incoming_ids)]
        expected_total = len(non_incoming_df) + len(incoming_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_manifest = Path(tmpdir) / "model_manifest.csv"
            shutil.copy2(real_manifest, temp_manifest)

            new_rows = [
                {
                    "model_id": model_id,
                    "model_family": "XLMR-large",
                    "training_cohort": "full_train_20k",
                    "correction_level": 1.0,
                    "seed": seed,
                    "features_dim": XLMR_HIDDEN_SIZE,
                    "optimal_threshold": 0.49,
                    "dev_macro_f1": 0.85,
                    "dev_accuracy": 0.86,
                    "dev_clickbait_f1": 0.82,
                    "test_seen_macro_f1": 0.84,
                    "test_seen_accuracy": 0.85,
                    "test_unseen_macro_f1": 0.81,
                    "test_unseen_accuracy": 0.82,
                    "model_path": f"models/xlmr/{model_id}",
                    "test_seen_preds": f"predictions/{model_id}_test_seen.csv",
                    "test_unseen_preds": f"predictions/{model_id}_test_unseen.csv",
                }
                for model_id, seed in [
                    ("xlmrlarge_full_train", 2026),
                    ("xlmrlarge_full_train_s42", 42),
                    ("xlmrlarge_full_train_s123", 123),
                ]
            ]
            combined = upsert_model_manifest(temp_manifest, new_rows)
            self.assertEqual(len(combined), expected_total)
            self.assertEqual(
                combined.loc[~combined["model_id"].astype(str).isin(incoming_ids), "model_id"].tolist(),
                non_incoming_df["model_id"].tolist(),
            )

            # Upsert again with modified value -> idempotent row count
            new_rows[0]["test_seen_macro_f1"] = 0.875
            combined2 = upsert_model_manifest(temp_manifest, new_rows)
            self.assertEqual(len(combined2), expected_total)
            updated_val = float(
                combined2.loc[combined2["model_id"] == "xlmrlarge_full_train", "test_seen_macro_f1"].iloc[0]
            )
            self.assertAlmostEqual(updated_val, 0.875)

    def test_preprocess_xlmr_with_replace_guard(self) -> None:
        tok = _DummyTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset.csv"
            pd.DataFrame({
                "record_id": [f"tr{i}" for i in range(4)] + [f"dv{i}" for i in range(2)] + ["ts0", "ts1", "tu0", "tu1"],
                "split": ["train"] * 4 + ["dev"] * 2 + ["test_seen"] * 2 + ["test_unseen"] * 2,
                "title": [f"Tiêu đề mẫu {i}!" for i in range(10)],
            }).to_csv(dataset_path, index=False)

            out_dir = root / "xlmr"
            rep_path = root / "preproc.json"
            report = preprocess_xlmr.run_preprocessing(
                dataset_path=dataset_path,
                output_dir=out_dir,
                report_path=rep_path,
                tokenizer=tok,
            )
            self.assertIn("train", report["splits"])
            self.assertTrue((out_dir / "preprocessor_config.json").exists())
            self.assertTrue(rep_path.exists())

            with self.assertRaises(FileExistsError):
                preprocess_xlmr.run_preprocessing(
                    dataset_path=dataset_path,
                    output_dir=out_dir,
                    report_path=rep_path,
                    tokenizer=tok,
                    replace=False,
                )

            report2 = preprocess_xlmr.run_preprocessing(
                dataset_path=dataset_path,
                output_dir=out_dir,
                report_path=rep_path,
                tokenizer=tok,
                replace=True,
            )
            self.assertEqual(report2["max_length"], report["max_length"])

    def test_tiny_xlmr_training_tuning_final_and_eval_smoke(self) -> None:
        tok = _DummyTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tokenized_dir = root / "tokenized"
            tokenized_dir.mkdir(parents=True)

            # Build tiny synthetic .npz splits
            for split_name, prefix, n in [
                ("train", "tr", 12),
                ("dev", "dv", 8),
                ("test_seen", "ts", 6),
                ("test_unseen", "tu", 6),
            ]:
                df = pd.DataFrame({
                    "record_id": [f"{prefix}{i}" for i in range(n)],
                    "title": [
                        f"SỐC clickbait tin nóng {i}!" if i % 2 == 1 else f"Bản tin thời sự chính thống {i}"
                        for i in range(n)
                    ],
                })
                payload, _ = tokenize_split(df, tok, max_length=16, batch_size=8)
                save_tokenized_split(
                    tokenized_dir / f"{split_name}.npz",
                    payload["record_id"],
                    payload["input_ids"],
                    payload["attention_mask"],
                    payload["lengths"],
                )

            train_labels_df = pd.DataFrame({
                "record_id": [f"tr{i}" for i in range(12)],
                "label": [i % 2 for i in range(12)],
            })
            dev_labels_df = pd.DataFrame({
                "record_id": [f"dv{i}" for i in range(8)],
                "label": [i % 2 for i in range(8)],
            })
            train_ds = build_tokenized_dataset(
                load_tokenized_split(tokenized_dir / "train.npz", labels_df=train_labels_df)
            )
            dev_ds = build_tokenized_dataset(
                load_tokenized_split(tokenized_dir / "dev.npz", labels_df=dev_labels_df)
            )

            # Direct helper test: balanced class_weight + checkpoint reload + inference
            ckpt_dir = root / "models" / "smoke_balanced"
            log_csv = root / "reports" / "smoke_log.csv"
            run_res = train_and_evaluate_run(
                model_or_name=_build_tiny_xlmr_model(),
                train_dataset=train_ds,
                dev_dataset=dev_ds,
                output_dir=ckpt_dir,
                lr=5e-4,
                epochs=2,
                batch_size=4,
                grad_accum=1,
                class_weight="balanced",
                eval_interval_epochs=1.0,
                precision="fp32",
                seed=2026,
                log_csv_path=log_csv,
            )
            self.assertTrue((ckpt_dir / "config.json").exists())
            self.assertTrue(log_csv.exists())
            self.assertIn("macro_f1", run_res["tuned_dev_metrics"])

            test_seen_ds = build_tokenized_dataset(load_tokenized_split(tokenized_dir / "test_seen.npz"))
            probs = predict_from_checkpoint(checkpoint_dir=ckpt_dir, dataset=test_seen_ds, batch_size=4, precision="fp32")
            self.assertEqual(probs.shape, (6,))
            self.assertFalse(np.isnan(probs).any())

            # End-to-end tune (with --resume) + final (2 seeds) + eval_xlmr (CPU)
            reports_dir = root / "reports" / "xlmr"
            models_dir = root / "models" / "xlmr"
            preds_dir = root / "predictions"

            best_cfg = train_xlmr.run_tuning(
                tokenized_dir=tokenized_dir,
                model_name="tiny-xlmr-smoke",
                reports_dir=reports_dir,
                models_dir=models_dir,
                train_dataset=train_ds,
                dev_dataset=dev_ds,
                tokenizer=None,
                epochs=1,
                batch_size=4,
                eval_interval=1.0,
                precision="fp32",
                tune_seed=2026,
                model_factory=_build_tiny_xlmr_model,
            )
            self.assertTrue((reports_dir / "xlmr_best_config.json").exists())
            self.assertEqual(len(pd.read_csv(reports_dir / "xlmr_tuning_results.csv")), 6)

            # Verify --resume skips completed configs cleanly
            best_cfg_resumed = train_xlmr.run_tuning(
                tokenized_dir=tokenized_dir,
                model_name="tiny-xlmr-smoke",
                reports_dir=reports_dir,
                models_dir=models_dir,
                train_dataset=train_ds,
                dev_dataset=dev_ds,
                tokenizer=None,
                epochs=1,
                batch_size=4,
                eval_interval=1.0,
                precision="fp32",
                tune_seed=2026,
                resume=True,
                model_factory=_build_tiny_xlmr_model,
            )
            self.assertEqual(best_cfg_resumed["config_id"], best_cfg["config_id"])
            self.assertEqual(len(pd.read_csv(reports_dir / "xlmr_tuning_results.csv")), 6)

            runs_dev_df = train_xlmr.run_final(
                tokenized_dir=tokenized_dir,
                model_name="tiny-xlmr-smoke",
                reports_dir=reports_dir,
                models_dir=models_dir,
                preds_dir=preds_dir,
                train_dataset=train_ds,
                dev_dataset=dev_ds,
                tokenizer=None,
                lr=float(best_cfg["lr"]),
                class_weight=best_cfg["class_weight"],
                seeds=[2026, 42],
                epochs=1,
                batch_size=4,
                eval_interval=1.0,
                precision="fp32",
                model_factory=_build_tiny_xlmr_model,
            )
            self.assertEqual(len(runs_dev_df), 2)
            self.assertTrue((preds_dir / "xlmrlarge_full_train_test_seen.csv").exists())
            self.assertTrue((preds_dir / "xlmrlarge_full_train_s42_test_unseen.csv").exists())

            # Fake an ML baseline prediction CSV + gold splits + existing manifest for eval_xlmr
            save_prediction_csv(
                [f"ts{i}" for i in range(6)],
                [0, 1, 0, 0, 1, 0],
                [0.2, 0.8, 0.3, 0.4, 0.7, 0.35],
                preds_dir / "logisticregression_full_train_test_seen.csv",
            )
            save_prediction_csv(
                [f"tu{i}" for i in range(6)],
                [0, 1, 0, 0, 1, 0],
                [0.2, 0.8, 0.3, 0.4, 0.7, 0.35],
                preds_dir / "logisticregression_full_train_test_unseen.csv",
            )
            seen_unlabeled = pd.DataFrame({"record_id": [f"ts{i}" for i in range(6)]})
            seen_gold = pd.DataFrame({"record_id": [f"ts{i}" for i in range(5)], "label": [0, 1, 0, 1, 0]})
            unseen_unlabeled = pd.DataFrame({"record_id": [f"tu{i}" for i in range(6)]})
            unseen_gold = pd.DataFrame({"record_id": [f"tu{i}" for i in range(5)], "label": [0, 1, 0, 1, 0]})

            manifest_path = root / "model_manifest.csv"
            shutil.copy2(PROJECT_ROOT / "data/manifests/model_manifest.csv", manifest_path)
            manifest_before = pd.read_csv(manifest_path)
            smoke_ids = {"xlmrlarge_full_train", "xlmrlarge_full_train_s42"}
            expected_manifest_total = int((~manifest_before["model_id"].astype(str).isin(smoke_ids)).sum()) + len(smoke_ids)

            eval_out = eval_xlmr.run_evaluation(
                runs_dev_csv=reports_dir / "xlmr_runs_dev.csv",
                preds_dir=preds_dir,
                reports_dir=reports_dir,
                manifest_path=manifest_path,
                tuning_csv=reports_dir / "xlmr_tuning_results.csv",
                n_bootstrap=50,
                baseline_ids=("logisticregression_full_train",),
                test_seen_unlabeled=seen_unlabeled,
                test_seen_gold=seen_gold,
                test_unseen_unlabeled=unseen_unlabeled,
                test_unseen_gold=unseen_gold,
            )
            self.assertEqual(eval_out["n_runs"], 2)
            self.assertEqual(eval_out["manifest_total_rows"], expected_manifest_total)
            self.assertTrue((reports_dir / "xlmr_runs.csv").exists())
            self.assertTrue((reports_dir / "xlmr_summary.csv").exists())
            self.assertTrue((reports_dir / "xlmr_bootstrap_vs_ml.csv").exists())
            self.assertTrue((reports_dir / "xlmr_manifest.csv").exists())
            report_text = Path(eval_out["report_md_path"]).read_text(encoding="utf-8")
            self.assertIn("Paired Bootstrap Comparison", report_text)
            self.assertIn("1.00e-05", report_text)
            boot_df = pd.read_csv(reports_dir / "xlmr_bootstrap_vs_ml.csv")
            self.assertEqual(len(boot_df), 2)  # seen + unseen vs logisticregression_full_train

            # Verify preprocessor_config.json schema compatibility check
            cfg_json = json.dumps({"model_name": "xlm-roberta-large", "max_length": 64})
            self.assertIn("max_length", cfg_json)

    def test_eval_xlmr_no_torch_or_transformers_import(self) -> None:
        code = (
            "import sys; sys.path.insert(0, r'scripts'); "
            "import eval_xlmr; "
            "assert 'torch' not in sys.modules, 'torch should not be imported'; "
            "assert 'transformers' not in sys.modules, 'transformers should not be imported'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
