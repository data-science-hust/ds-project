# Shared Data Loader & Label Correction Module

This directory (`src/ds_project/correction/`) contains the shared data loading utilities, label correction logic, and dataset interfaces for the Vietnamese Clickbait Detection project (Group 5).

---

## Table of Contents
1. [Overview](#1-overview)
2. [Dataset Splits & Label Semantics](#2-dataset-splits--label-semantics)
3. [Loader API Reference (`loader.py`)](#3-loader-api-reference-loaderpy)
   - [`load_train_set`](#load_train_set)
   - [`load_subset`](#load_subset)
   - [`load_dev`](#load_dev)
   - [`load_test`](#load_test)
   - [`get_train_dev_matrices`](#get_train_dev_matrices)
4. [Label Correction & Manifests Reference](#4-label-correction--manifests-reference)
   - [`correction_manifest.csv`](#correction_manifestcsv)
   - [`subset_label_versions.csv`](#subset_label_versionscsv)
5. [Feature Pipeline Integration](#5-feature-pipeline-integration)
   - [Input Variants (v1, v2, v3)](#input-variants-v1-v2-v3)
6. [Code Recipes for Downstream Models](#6-code-recipes-for-downstream-models)
   - [Machine Learning (LR, SVM, RF, XGBoost)](#recipe-1-machine-learning-training)
   - [Deep Learning / Transformers (PhoBERT, XLM-R)](#recipe-2-deep-learning--transformers-training)
   - [Correction-Scaling Ablation Study](#recipe-3-correction-scaling-ablation-study)
7. [Testing & Verification](#7-testing--verification)

---

## 1. Overview

Stage 4 of the project establishes a unified, reproducible dataset contract across all modeling workstreams:
- **Full Train Set (`train_full_corrected.csv`):** 20,000 articles where the 6,400-article training subset has been updated with Human Gold labels, and the remaining 13,600 articles retain LLM labels.
- **Nested Correction Ablations:** Slices of the 6,400 subset with progressive correction levels (0%, 25%, 50%, 75%, 100% of disagreements corrected) across 3 fixed random seeds (`2026`, `42`, `123`).
- **Feature Pipelines:** Unified support for baseline TF-IDF (v1), 25 Warsaw stylometric features (v2), and early fusion hybrid (v3).

---

## 2. Dataset Splits & Label Semantics

| Split | Total Records | Label Source | Description |
| :--- | ---: | :--- | :--- |
| `train` (full) | 20,000 | Human Gold (subset) + LLM (outside) | The canonical training set for all final models. |
| `train` (subset) | 6,400 | Human Gold (target) vs LLM (initial) | Designated cohort for correction scaling ablation. |
| `dev` | 1,400 | Human Gold (1,395 adjudicated) | Validation set for model selection, threshold tuning, and variant comparison. |
| `test_seen` | 1,000 | Held-out blind | Final evaluation on the 8 seen news outlets. |
| `test_unseen` | 1,200 | Held-out blind | Final evaluation on the 4 unseen outlets (`saostar`, `yan`, `vietnamnet`, `nhandan`). |

### Label Encoding
- `0`: Non-clickbait headline (informative, objective, key facts disclosed).
- `1`: Clickbait headline (curiosity gap, emotional exaggeration, dangling questions).

---

## 3. Loader API Reference (`loader.py`)

Import directly from `ds_project.correction.loader`:

```python
from ds_project.correction.loader import (
    load_train_set,
    load_subset,
    load_dev,
    load_test,
    get_train_dev_matrices,
)
```

### `load_train_set`
Loads all 20,000 training records.

```python
def load_train_set(
    variant: Literal["full_corrected", "llm_raw"] = "full_corrected",
    train_corrected_path: Path | str | None = None,
) -> pd.DataFrame
```

- **Parameters:**
  - `variant`: `"full_corrected"` (default) replaces LLM labels with Human Gold inside the 6,400 subset. `"llm_raw"` loads original raw LLM labels across all 20,000 records.
- **Returns:** `pd.DataFrame` with 20,000 rows. Includes columns: `record_id`, `title`, `sapo`, `category`, `source`, `label`, `is_corrected`, `llm_label`, `gold_label`.

### `load_subset`
Loads the 6,400 training subset records with labels at a specific correction level and seed.

```python
def load_subset(
    correction_level: float = 1.0,
    seed: int = 2026,
    subset_versions_path: Path | str | None = None,
) -> pd.DataFrame
```

- **Parameters:**
  - `correction_level`: Float in `{0.0, 0.25, 0.50, 0.75, 1.0}`.
  - `seed`: Random seed for nested disagreement permutation (`2026`, `42`, or `123`).
- **Returns:** `pd.DataFrame` with 6,400 rows and the resulting `label` column.

### `load_dev`
Loads the 1,400 dev set records.

```python
def load_dev(include_gold: bool = True) -> pd.DataFrame
```

- **Parameters:**
  - `include_gold`: If `True`, merges human adjudicated labels from `dev_gold.csv` and drops unadjudicated exclusions (yields 1,395 clean rows).
- **Returns:** `pd.DataFrame` with columns `record_id`, `title`, `sapo`, `category`, `source`, `label`.

### `load_test`
Loads held-out blind test records.

```python
def load_test(kind: Literal["seen", "unseen"] = "seen") -> pd.DataFrame
```

- **Parameters:**
  - `kind`: `"seen"` returns 1,000 records from seen outlets; `"unseen"` returns 1,200 records from unseen outlets.
- **Returns:** `pd.DataFrame` with metadata and input text (`record_id`, `title`, `sapo`, `category`, `source`).

### `get_train_dev_matrices`
High-level helper returning transformed feature matrices and label vectors ready for ML classifiers.

```python
def get_train_dev_matrices(
    train_variant: Literal["full_corrected", "llm_raw"] = "full_corrected",
    input_variant: Literal["v1", "v2", "v3"] = "v3",
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, ClickbaitFeaturePipeline]
```

- **Returns:** `(X_train, y_train, X_dev, y_dev, pipeline)`.

---

## 4. Label Correction & Manifests Reference

All manifest outputs are stored in `data/manifests/`.

### `correction_manifest.csv`
Provides a full audit trail for each record across seeds and correction levels.

| Field | Type | Description |
| :--- | :--- | :--- |
| `record_id` | string | Unique article identifier. |
| `seed` | int | Random seed (`2026`, `42`, `123`). |
| `correction_level` | float | Fraction of disagreements corrected (`0.0`, `0.25`, `0.5`, `0.75`, `1.0`). |
| `label_before` | int | Original LLM label. |
| `label_after` | int | Label assigned after applying the correction step. |
| `was_corrected` | bool | `True` if this record was selected and had its label changed. |

### `subset_label_versions.csv`
A compact lookup table for fast merging into training scripts.
- **Fields:** `record_id`, `seed`, `correction_level`, `label`.

---

## 5. Feature Pipeline Integration

The feature engineering package is located at `src/ds_project/features/`.

### Input Variants (v1, v2, v3)

```python
from ds_project.features.pipeline import ClickbaitFeaturePipeline

# Version 1: Word + Char TF-IDF baseline (35,000 dimensions)
pipe_v1 = ClickbaitFeaturePipeline(variant="v1")

# Version 2: 25 Warsaw Stylometric Features (25 dimensions, scaled)
pipe_v2 = ClickbaitFeaturePipeline(variant="v2")

# Version 3: Early Fusion Concat(TF-IDF, Warsaw 25) (35,025 dimensions)
pipe_v3 = ClickbaitFeaturePipeline(variant="v3")
```

Important rules:
1. **Train-only fitting:** Always `fit` or `fit_transform` on the training dataframe (`df_train`). Never fit on `df_dev` or `df_test`.
2. **Text Normalization:** Text is automatically normalized to canonical Unicode NFC while preserving uppercase words and punctuation marks (`!`, `?`, `...`, `“...”`).

---

## 6. Code Recipes for Downstream Models

### Recipe 1: Machine Learning Training
Applicable to Logistic Regression, Linear SVM, Random Forest, XGBoost, LightGBM:

```python
from ds_project.correction.loader import load_train_set, load_dev
from ds_project.features.pipeline import ClickbaitFeaturePipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 1. Load data
df_train = load_train_set(variant="full_corrected")
df_dev = load_dev()

# 2. Extract features using Version 3
pipeline = ClickbaitFeaturePipeline(variant="v3")
X_train = pipeline.fit_transform(df_train)
y_train = df_train["label"].to_numpy()

X_dev = pipeline.transform(df_dev)
y_dev = df_dev["label"].to_numpy()

# 3. Fit classifier
clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=2026)
clf.fit(X_train, y_train)

# 4. Evaluate on Dev
y_pred = clf.predict(X_dev)
print(classification_report(y_dev, y_pred, target_names=["non-clickbait", "clickbait"]))
```

### Recipe 2: Deep Learning / Transformers Training
Applicable to PhoBERT, XLM-R, TextCNN, BiLSTM:

```python
from ds_project.correction.loader import load_train_set, load_dev
from transformers import AutoTokenizer

# Load full train set
df_train = load_train_set(variant="full_corrected")
df_dev = load_dev()

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Tokenize titles
train_encodings = tokenizer(
    df_train["title"].tolist(),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)
train_labels = df_train["label"].tolist()
```

### Recipe 3: Correction-Scaling Ablation Study
Measure model performance across 0%, 25%, 50%, 75%, and 100% human correction:

```python
from ds_project.correction.loader import load_subset, load_dev
from ds_project.features.pipeline import ClickbaitFeaturePipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df_dev = load_dev()
levels = [0.0, 0.25, 0.50, 0.75, 1.0]
seeds = [2026, 42, 123]

for seed in seeds:
    for lvl in levels:
        df_sub = load_subset(correction_level=lvl, seed=seed)
        pipe = ClickbaitFeaturePipeline(variant="v3")
        X_sub = pipe.fit_transform(df_sub)
        y_sub = df_sub["label"].to_numpy()

        X_dev = pipe.transform(df_dev)
        y_dev = df_dev["label"].to_numpy()

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        model.fit(X_sub, y_sub)
        pred = model.predict(X_dev)
        f1 = f1_score(y_dev, pred, average="macro")
        print(f"Seed {seed} | Level {lvl:.0%} -> Dev Macro-F1: {f1:.4f}")
```

---

## 7. Testing & Verification

Unit tests covering this module and related components are available in `tests/`:
- `tests/test_correction_and_loader.py`: Verifies split loading, dataset length contracts, and strict nested inclusion of corrected samples across levels.
- `tests/test_features_pipeline.py`: Verifies v1, v2, and v3 matrix shapes and scikit-learn transformer compliance.
- `tests/test_stylometric.py`: Verifies all 25 Warsaw features on edge cases.
- `tests/test_eda.py`: Verifies analytical routines.

Execute the test suite:
```powershell
python -m unittest discover -s tests -v
```
