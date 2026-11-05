# Label Versions and Correction Manifest Report

## 1. Full Train Set After Correction Summary

- Total records: 20,000
- In subset (human gold): 6,400
- Outside subset (LLM labels): 13,600
- Final clickbait labels (class 1): 7,798 (38.99%)
- Final non-clickbait labels (class 0): 12,202 (61.01%)
- Total labels corrected from LLM on subset: 1,325 (6.62%)

## 2. Nested Correction Ablation Levels on Training Subset

| Seed | Correction Level | Target Ratio | Corrected Disagreements | Total Subset Records |
| ---: | ---: | ---: | ---: | ---: |
| 2026 | 0% | 0.00 | 0 | 6,400 |
| 2026 | 25% | 0.25 | 331 | 6,400 |
| 2026 | 50% | 0.50 | 662 | 6,400 |
| 2026 | 75% | 0.75 | 994 | 6,400 |
| 2026 | 100% | 1.00 | 1,325 | 6,400 |
| 42 | 0% | 0.00 | 0 | 6,400 |
| 42 | 25% | 0.25 | 331 | 6,400 |
| 42 | 50% | 0.50 | 662 | 6,400 |
| 42 | 75% | 0.75 | 994 | 6,400 |
| 42 | 100% | 1.00 | 1,325 | 6,400 |
| 123 | 0% | 0.00 | 0 | 6,400 |
| 123 | 25% | 0.25 | 331 | 6,400 |
| 123 | 50% | 0.50 | 662 | 6,400 |
| 123 | 75% | 0.75 | 994 | 6,400 |
| 123 | 100% | 1.00 | 1,325 | 6,400 |

## 3. Outputs and Deliverables

- `data/processed/train_full_corrected.csv`: 20,000 train records with final corrected labels.
- `data/manifests/correction_manifest.csv`: Audit trail per record, seed, and correction level.
- `data/manifests/subset_label_versions.csv`: Label lookup per record, seed, and correction level.
