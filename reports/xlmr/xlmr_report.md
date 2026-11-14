# XLM-RoBERTa-large Evaluation Report

## 1. Model & Training Configuration
- **Model:** `xlm-roberta-large` (`features_dim=1024`)
- **Training cohort:** `full_train_20k` (`train_full_corrected.csv`, 20,000 rows)
- **Learning rate:** `3e-05` | **Class weight:** `balanced` | **Epochs:** `5` | **Effective batch size:** `32`

- **Preprocessing `max_length`:** `64` (selected on Train + Dev only)

| split | count | min | median | p95 | p99 | max | truncation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 20000 | 5 | 22.0000 | 36.0000 | 42.0000 | 62 | 0.0000 |
| dev | 1400 | 7 | 22.0000 | 36.0000 | 41.0000 | 48 | 0.0000 |
| test_seen | 1000 | 8 | 22.0000 | 36.0000 | 42.0000 | 49 | 0.0000 |
| test_unseen | 1200 | 5 | 19.0000 | 28.0000 | 31.0100 | 37 | 0.0000 |

## 2. Dev Hyperparameter Tuning

| config_id | lr | class_weight | best_epoch | dev_macro_f1_050 | optimal_threshold | dev_macro_f1 | dev_accuracy | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr3e-5_cw_balanced | 3.00e-05 | balanced | 0.4992 | 0.8398 | 0.5200 | 0.8419 | 0.8423 | 460.4500 |
| lr1e-5_cw_none | 1.00e-05 | none | 0.4992 | 0.8364 | 0.5100 | 0.8371 | 0.8373 | 515.0700 |
| lr1e-5_cw_balanced | 1.00e-05 | balanced | 1.4992 | 0.8264 | 0.3500 | 0.8354 | 0.8358 | 473.2700 |
| lr3e-5_cw_none | 3.00e-05 | none | 0.4992 | 0.8318 | 0.5000 | 0.8318 | 0.8323 | 460.7300 |
| lr2e-5_cw_balanced | 2.00e-05 | balanced | 2.0000 | 0.8261 | 0.4600 | 0.8306 | 0.8308 | 491.8000 |
| lr2e-5_cw_none | 2.00e-05 | none | 0.4992 | 0.8300 | 0.4400 | 0.8301 | 0.8301 | 466.0900 |

## 3. Per-Seed Results (Dev, Test Seen, Test Unseen)

| model_id | seed | best_epoch | optimal_threshold | dev_macro_f1 | dev_accuracy | test_seen_macro_f1 | test_seen_accuracy | test_seen_precision_1 | test_seen_recall_1 | test_seen_f1_1 | test_unseen_macro_f1 | test_unseen_accuracy | test_unseen_precision_1 | test_unseen_recall_1 | test_unseen_f1_1 | seen_unseen_macro_f1_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xlmrlarge_full_train | 2026 | 0.4992 | 0.5200 | 0.8419 | 0.8423 | 0.8262 | 0.8277 | 0.7635 | 0.8638 | 0.8106 | 0.7965 | 0.8476 | 0.6106 | 0.8054 | 0.6946 | 0.0297 |
| xlmrlarge_full_train_s42 | 42 | 2.9984 | 0.3700 | 0.8454 | 0.8459 | 0.8207 | 0.8216 | 0.7480 | 0.8779 | 0.8078 | 0.8083 | 0.8601 | 0.6424 | 0.7899 | 0.7086 | 0.0124 |
| xlmrlarge_full_train_s123 | 123 | 1.4992 | 0.4600 | 0.8285 | 0.8287 | 0.8128 | 0.8136 | 0.7372 | 0.8756 | 0.8004 | 0.7695 | 0.8258 | 0.5706 | 0.7704 | 0.6556 | 0.0433 |

## 4. Confusion Matrices (Test Splits)

| model_id | test_seen_tn | test_seen_fp | test_seen_fn | test_seen_tp | test_unseen_tn | test_unseen_fp | test_unseen_fn | test_unseen_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xlmrlarge_full_train | 458 | 114 | 58 | 368 | 805 | 132 | 50 | 207 |
| xlmrlarge_full_train_s42 | 446 | 126 | 52 | 374 | 824 | 113 | 54 | 203 |
| xlmrlarge_full_train_s123 | 439 | 133 | 53 | 373 | 788 | 149 | 59 | 198 |

## 5. Multi-Seed Summary (Mean ± Std)

| metric | n_seeds | mean | std | min | max | mean_plus_minus_std |
| --- | --- | --- | --- | --- | --- | --- |
| optimal_threshold | 3 | 0.4500 | 0.0755 | 0.3700 | 0.5200 | 0.4500 ± 0.0755 |
| dev_macro_f1 | 3 | 0.8386 | 0.0089 | 0.8285 | 0.8454 | 0.8386 ± 0.0089 |
| dev_accuracy | 3 | 0.8389 | 0.0091 | 0.8287 | 0.8459 | 0.8389 ± 0.0091 |
| dev_f1_0 | 3 | 0.8463 | 0.0104 | 0.8346 | 0.8542 | 0.8463 ± 0.0104 |
| dev_f1_1 | 3 | 0.8308 | 0.0075 | 0.8223 | 0.8365 | 0.8308 ± 0.0075 |
| test_seen_macro_f1 | 3 | 0.8199 | 0.0068 | 0.8128 | 0.8262 | 0.8199 ± 0.0068 |
| test_seen_accuracy | 3 | 0.8210 | 0.0070 | 0.8136 | 0.8277 | 0.8210 ± 0.0070 |
| test_seen_precision_1 | 3 | 0.7495 | 0.0132 | 0.7372 | 0.7635 | 0.7495 ± 0.0132 |
| test_seen_recall_1 | 3 | 0.8725 | 0.0075 | 0.8638 | 0.8779 | 0.8725 ± 0.0075 |
| test_seen_f1_1 | 3 | 0.8063 | 0.0052 | 0.8004 | 0.8106 | 0.8063 ± 0.0052 |
| test_unseen_macro_f1 | 3 | 0.7914 | 0.0199 | 0.7695 | 0.8083 | 0.7914 ± 0.0199 |
| test_unseen_accuracy | 3 | 0.8445 | 0.0174 | 0.8258 | 0.8601 | 0.8445 ± 0.0174 |
| test_unseen_precision_1 | 3 | 0.6079 | 0.0360 | 0.5706 | 0.6424 | 0.6079 ± 0.0360 |
| test_unseen_recall_1 | 3 | 0.7886 | 0.0175 | 0.7704 | 0.8054 | 0.7886 ± 0.0175 |
| test_unseen_f1_1 | 3 | 0.6863 | 0.0274 | 0.6556 | 0.7086 | 0.6863 ± 0.0274 |
| seen_unseen_macro_f1_gap | 3 | 0.0285 | 0.0155 | 0.0124 | 0.0433 | 0.0285 ± 0.0155 |
| seen_unseen_accuracy_gap | 3 | -0.0235 | 0.0135 | -0.0385 | -0.0122 | -0.0235 ± 0.0135 |

## 6. Paired Bootstrap Comparison vs Full-Train ML Baselines (95% CI)

| split | xlmr_model_id | baseline_model_id | n_common | macro_f1_baseline | macro_f1_xlmr | delta_macro_f1 | ci_low | ci_high | significant_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_seen | xlmrlarge_full_train | logisticregression_full_train | 998 | 0.7968 | 0.8262 | 0.0294 | 0.0067 | 0.0525 | True |
| test_seen | xlmrlarge_full_train | linearsvm_full_train | 998 | 0.7754 | 0.8262 | 0.0509 | 0.0269 | 0.0743 | True |
| test_seen | xlmrlarge_full_train | randomforest_full_train | 998 | 0.7722 | 0.8262 | 0.0540 | 0.0283 | 0.0792 | True |
| test_seen | xlmrlarge_full_train | xgboost_full_train | 998 | 0.7793 | 0.8262 | 0.0469 | 0.0223 | 0.0716 | True |
| test_unseen | xlmrlarge_full_train | logisticregression_full_train | 1194 | 0.7053 | 0.7965 | 0.0913 | 0.0622 | 0.1195 | True |
| test_unseen | xlmrlarge_full_train | linearsvm_full_train | 1194 | 0.7077 | 0.7965 | 0.0888 | 0.0601 | 0.1183 | True |
| test_unseen | xlmrlarge_full_train | randomforest_full_train | 1194 | 0.6897 | 0.7965 | 0.1068 | 0.0747 | 0.1424 | True |
| test_unseen | xlmrlarge_full_train | xgboost_full_train | 1194 | 0.6916 | 0.7965 | 0.1049 | 0.0747 | 0.1372 | True |

## 7. Manifest Rows Registered

| model_id | seed | optimal_threshold | dev_macro_f1 | test_seen_macro_f1 | test_unseen_macro_f1 | model_path |
| --- | --- | --- | --- | --- | --- | --- |
| xlmrlarge_full_train | 2026 | 0.5200 | 0.8419 | 0.8262 | 0.7965 | models/xlmr/xlmrlarge_full_train |
| xlmrlarge_full_train_s42 | 42 | 0.3700 | 0.8454 | 0.8207 | 0.8083 | models/xlmr/xlmrlarge_full_train_s42 |
| xlmrlarge_full_train_s123 | 123 | 0.4600 | 0.8285 | 0.8128 | 0.7695 | models/xlmr/xlmrlarge_full_train_s123 |
