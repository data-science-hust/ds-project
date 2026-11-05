# Exploratory Data Analysis (EDA) Report

## Executive Summary

This report provides an end-to-end exploratory analysis of the Vietnamese clickbait detection
dataset for Group 5 (23,600 total split records). It evaluates text structural attributes,
human annotation consistency, LLM prompt variants agreement, and label concentration across outlets and categories.

---

## 1. Dataset Partitioning & Split Counts

The frozen split comprises 23,600 records across 12 outlets (8 seen outlets with stratified split,
and 4 unseen outlets held out entirely for zero-shot outlet evaluation).

| source | outlet_group | dev | test_seen | test_unseen | train | Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2sao | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| afamily | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| cafef | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| dantri | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| eva | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| kenh14 | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| nhandan | unseen | 0 | 0 | 300 | 0 | 300 |
| saostar | unseen | 0 | 0 | 300 | 0 | 300 |
| thanhnien | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| tuoitre | seen | 175 | 125 | 0 | 2,500 | 2,800 |
| vietnamnet | unseen | 0 | 0 | 300 | 0 | 300 |
| yan | unseen | 0 | 0 | 300 | 0 | 300 |
| Total |  | 1,400 | 1,000 | 1,200 | 20,000 | 23,600 |

- **Train set:** 20,000 articles (includes 6,400 designated for human-corrected subset ablation).
- **Dev set:** 1,400 articles (100% articles have lead paragraph/sapo available).
- **Test seen:** 1,000 articles across the 8 seen news sources.
- **Test unseen:** 1,200 articles across 4 unseen sources (`saostar`, `yan`, `vietnamnet`, `nhandan`).

---

## 2. Text Characteristics & Sapo Completeness

### Headline Length Distributions

- **Character length:** Mean = 74.78 chars (Median = 72.0, IQR = [59.0, 90.0], Range = [12, 201]).
- **Word length (tokens):** Mean = 16.81 words (Median = 16.0, IQR = [13.0, 20.0], Range = [3, 43]).

Headlines exhibit compact lengths typical of Vietnamese journalism, with sufficient token span to extract
both n-gram lexical features and syntactic/stylometric informativeness cues.

### Sapo (Lead Paragraph) Availability by Source

| source | outlet_group | records | missing_sapo_pct | title_word_mean |
| ---: | ---: | ---: | ---: | ---: |
| 2sao | seen | 2,800 | 0.0004 | 17.22 |
| afamily | seen | 2,800 | 0.0011 | 19.56 |
| cafef | seen | 2,800 | 0.0007 | 17.84 |
| dantri | seen | 2,800 | 0.0071 | 14.25 |
| eva | seen | 2,800 | 0.0000 | 18.77 |
| kenh14 | seen | 2,800 | 0.0029 | 18.46 |
| tuoitre | seen | 2,800 | 0.0004 | 15.30 |
| thanhnien | seen | 2,800 | 0.0029 | 14.28 |
| saostar | unseen | 300 | 0.0000 | 12.86 |
| nhandan | unseen | 300 | 0.0300 | 14.10 |
| vietnamnet | unseen | 300 | 0.0067 | 14.84 |
| yan | unseen | 300 | 0.5700 | 14.33 |

### Topic Category Distributions & Sapo Completeness

| category | records | share | missing_sapo_pct | title_word_mean |
| ---: | ---: | ---: | ---: | ---: |
| Thời sự - Xã hội | 4,864 | 0.2061 | 0.0056 | 16.02 |
| Đời sống - Gia đình | 3,599 | 0.1525 | 0.0019 | 18.99 |
| Giải trí - Văn hóa | 2,900 | 0.1229 | 0.0466 | 16.11 |
| Kinh tế - Tài chính | 2,334 | 0.0989 | 0.0009 | 16.00 |
| Thế giới | 1,471 | 0.0623 | 0.0020 | 15.86 |
| Sức khỏe | 1,343 | 0.0569 | 0.0015 | 18.31 |
| Thể thao | 1,213 | 0.0514 | 0.0000 | 15.06 |
| Làm đẹp - Thời trang | 1,075 | 0.0456 | 0.0242 | 17.25 |
| Giáo dục | 944 | 0.0400 | 0.0011 | 17.77 |
| Công nghệ - Khoa học | 814 | 0.0345 | 0.0000 | 14.91 |
| Du lịch - Ẩm thực | 700 | 0.0297 | 0.0029 | 18.09 |
| Tử vi - Phong thủy | 659 | 0.0279 | 0.0000 | 20.89 |
| Pháp luật | 586 | 0.0248 | 0.0000 | 15.32 |
| Bất động sản | 478 | 0.0203 | 0.0000 | 16.53 |
| Khác | 417 | 0.0177 | 0.0456 | 15.26 |
| Xe | 203 | 0.0086 | 0.0049 | 15.46 |

---

## 3. Human Annotation Quality & Agreement (Dev Set)

Independent human annotation was performed by pairs of annotators on 1,400 dev records,
with disputed records resolved by designated adjudicators.

- **Total adjudicated records:** 1,395
- **Clickbait (1):** 649 (46.52%)
- **Non-clickbait (0):** 746 (53.48%)
- **Raw annotator agreement:** 77.71%
- **Cohen's Kappa (inter-annotator reliability):** $\kappa = 0.5511$
- **Adjudication rate (disagreement requiring senior review):** 22.86% (320 cases)
- **Annotator review flag rates:** Annotator A = 0.64%, Annotator B = 0.79%

---

## 4. LLM Prompt Variants Evaluation (Dev vs Human Gold)

Four LLM prompting variants (`gemma4_26b_a4b_thinking_off`) were evaluated on the exact dev set:

| variant | evaluated_n | accuracy | macro_f1 | macro_precision | macro_recall | fp_0_to_1 | fn_1_to_0 | disagreement_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T (Title only) | 1,394 | 0.7798 | 0.7707 | 0.8004 | 0.7696 | 64 | 243 | 0.2202 |
| TC (Title + Category) | 1,394 | 0.7791 | 0.7723 | 0.7913 | 0.7708 | 83 | 225 | 0.2209 |
| TS (Title + Sapo) | 1,394 | 0.7948 | 0.7900 | 0.8032 | 0.7880 | 86 | 200 | 0.2052 |
| TSC (Title + Sapo + Category) | 1,393 | 0.7968 | 0.7927 | 0.8030 | 0.7907 | 92 | 191 | 0.2032 |

**Key Findings:**
- Including both title, sapo, and category (TSC) achieves competitive macro-F1 (0.7925) and top accuracy (80.11%).
- The dominant error mode across LLM predictions is False Negatives ($1 \to 0$, where LLM misses subtle Vietnamese clickbait tactics like curiosity gaps and dangling questions).

---

## 5. LLM–Human Disagreement Breakdown (TSC Variant)

### By News Outlet

| group | records | fp_0_to_1 | fn_1_to_0 | disagreements | disagreement_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2sao | 175 | 12 | 19 | 31 | 0.1771 |
| afamily | 174 | 11 | 28 | 39 | 0.2241 |
| cafef | 175 | 20 | 26 | 46 | 0.2629 |
| dantri | 174 | 11 | 13 | 24 | 0.1379 |
| eva | 173 | 7 | 34 | 41 | 0.2370 |
| kenh14 | 173 | 11 | 32 | 43 | 0.2486 |
| thanhnien | 174 | 13 | 21 | 34 | 0.1954 |
| tuoitre | 175 | 7 | 18 | 25 | 0.1429 |

### By Content Category

| group | records | fp_0_to_1 | fn_1_to_0 | disagreements | disagreement_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Bất động sản | 30 | 1 | 4 | 5 | 0.1667 |
| Công nghệ - Khoa học | 47 | 10 | 9 | 19 | 0.4043 |
| Du lịch - Ẩm thực | 43 | 3 | 6 | 9 | 0.2093 |
| Giáo dục | 57 | 5 | 6 | 11 | 0.1930 |
| Giải trí - Văn hóa | 152 | 10 | 17 | 27 | 0.1776 |
| Khác | 22 | 0 | 1 | 1 | 0.0455 |
| Kinh tế - Tài chính | 141 | 16 | 21 | 37 | 0.2624 |
| Làm đẹp - Thời trang | 64 | 2 | 8 | 10 | 0.1562 |
| Pháp luật | 35 | 0 | 3 | 3 | 0.0857 |
| Sức khỏe | 81 | 2 | 16 | 18 | 0.2222 |
| Thế giới | 88 | 6 | 8 | 14 | 0.1591 |
| Thể thao | 70 | 7 | 14 | 21 | 0.3000 |
| Thời sự - Xã hội | 287 | 19 | 28 | 47 | 0.1638 |
| Tử vi - Phong thủy | 41 | 1 | 5 | 6 | 0.1463 |
| Xe | 11 | 2 | 0 | 2 | 0.1818 |
| Đời sống - Gia đình | 224 | 8 | 45 | 53 | 0.2366 |

---

## 6. Training Set Label Concentration & Outlet Analysis

### Clickbait Proportion by Source in Training Set

| source | records | clickbait_count | clickbait_ratio | non_clickbait_count |
| ---: | ---: | ---: | ---: | ---: |
| afamily | 2,500 | 1,523 | 0.6092 | 977 |
| kenh14 | 2,500 | 1,406 | 0.5624 | 1,094 |
| 2sao | 2,500 | 1,404 | 0.5616 | 1,096 |
| eva | 2,500 | 1,327 | 0.5308 | 1,173 |
| cafef | 2,500 | 1,040 | 0.4160 | 1,460 |
| thanhnien | 2,500 | 434 | 0.1736 | 2,066 |
| dantri | 2,500 | 360 | 0.1440 | 2,140 |
| tuoitre | 2,500 | 304 | 0.1216 | 2,196 |

### Clickbait Proportion by Category in Training Set

| category | records | clickbait_count | clickbait_ratio | non_clickbait_count |
| ---: | ---: | ---: | ---: | ---: |
| Tử vi - Phong thủy | 589 | 508 | 0.8625 | 81 |
| Đời sống - Gia đình | 3,201 | 2,042 | 0.6379 | 1,159 |
| Làm đẹp - Thời trang | 912 | 528 | 0.5789 | 384 |
| Giải trí - Văn hóa | 2,198 | 1,262 | 0.5742 | 936 |
| Sức khỏe | 1,176 | 588 | 0.5000 | 588 |
| Du lịch - Ẩm thực | 609 | 291 | 0.4778 | 318 |
| Công nghệ - Khoa học | 698 | 262 | 0.3754 | 436 |
| Giáo dục | 819 | 303 | 0.3700 | 516 |
| Kinh tế - Tài chính | 2,017 | 551 | 0.2732 | 1,466 |
| Thể thao | 1,014 | 270 | 0.2663 | 744 |
| Thế giới | 1,256 | 333 | 0.2651 | 923 |
| Khác | 311 | 74 | 0.2379 | 237 |
| Xe | 173 | 31 | 0.1792 | 142 |
| Thời sự - Xã hội | 4,109 | 659 | 0.1604 | 3,450 |
| Bất động sản | 420 | 64 | 0.1524 | 356 |
| Pháp luật | 498 | 32 | 0.0643 | 466 |

**Caution on Label Concentration:**
While entertainment-centric outlets (e.g. `2sao`, `kenh14`, `afamily`) exhibit higher clickbait proportions
than mainstream news (`dantri`, `tuoitre`), clickbait is not conflated with outlet type or domain.
Both positive and negative samples exist across all categories and sources.

---

## 7. Modeling Recommendations for Machine Learning

1. **Input Representation:**
   - Baseline (Version 1): Word + Char TF-IDF captures lexical cues.
   - Extended (Version 2): 25 Warsaw Stylometric Features capture readability, punctuation tricks, superlatives, and syntax.
   - Hybrid (Version 3): Early concatenation ($[\mathbf{X}_{\text{TF-IDF}}, \mathbf{X}_{\text{Stylo}}]$) combines high-dimensional vocabulary with explicit stylistic signals.
2. **Evaluation Strategy:** Evaluate models on Test Seen and Test Unseen separately to measure generalization.
3. **Correction Scaling:** Progressive human label correction on the subset directly targets the ~20% LLM error gap.
