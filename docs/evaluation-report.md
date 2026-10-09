# Evaluation Report

> **Validation results — not final test results.**  
> The official test split (`data/raw/test.tsv`) remains strictly untouched for final evaluation in later steps. This initial experiment establishes the classical machine learning baseline.

---

# Classical ML Baseline — Initial Experiment

## Overview
- **Dataset:** LIAR Benchmark (ACL 2017)
- **Training Split Size:** 10,269 rows (`data/raw/train.tsv`)
- **Validation Split Size:** 1,284 rows (`data/raw/valid.tsv`)
- **Test Set Status:** Strictly untouched (0 test rows evaluated or accessed)
- **Target Classes:** 3 mapped project labels (`likely_reliable`, `misleading`, `likely_fake`)

## Pipeline Architecture & Configuration

1. **Feature Representation:** `TfidfVectorizer`
   - `input`: `clean_statement`
   - `lowercase`: `True`
   - `ngram_range`: `(1, 2)` (unigrams and bigrams)
   - `min_df`: `2`
   - `max_df`: `0.95`
   - `max_features`: `50,000`
   - `sublinear_tf`: `True`

2. **Classifier:** `LogisticRegression`
   - `C`: `1.0`
   - `max_iter`: `1000`
   - `class_weight`: `None`
   - `solver`: `lbfgs`
   - `random_state`: `42`

---

## Validation Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **45.64%** (0.4564) |
| **Macro Precision** | **0.4556** |
| **Macro Recall** | **0.4486** |
| **Macro F1** | **0.4465** |
| **Weighted F1** | **0.4507** |

---

## Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support (Actual Count) |
| :--- | :--- | :--- | :--- | :--- |
| **likely_fake** | 0.4462 | 0.3061 | 0.3631 | 379 |
| **likely_reliable** | 0.4753 | 0.5262 | 0.4994 | 420 |
| **misleading** | 0.4454 | 0.5134 | 0.4770 | 485 |
| **Total / Macro Avg** | **0.4556** | **0.4486** | **0.4465** | **1,284** |

---

## Confusion Matrix

*Learned Class Order:* `['likely_fake', 'likely_reliable', 'misleading']`  
*Format: Rows = Actual Ground Truth, Columns = Model Prediction*

| Actual \ Predicted | predicted: `likely_fake` | predicted: `likely_reliable` | predicted: `misleading` | Total |
| :--- | :--- | :--- | :--- | :--- |
| **actual: likely_fake** | **116** | 101 | 162 | 379 |
| **actual: likely_reliable** | 51 | **221** | 148 | 420 |
| **actual: misleading** | 93 | 143 | **249** | 485 |
| **Total Predicted** | 260 | 465 | 559 | 1,284 |

---

## Leakage-Sensitivity Diagnostic (Overlap Excluded)

A sensitivity diagnostic was evaluated on the validation set after excluding the 5 statements that appear verbatim in the training set:
- **Excluded Statements:** 5 statements (100% of train-validation overlaps)
- **Evaluated Validation Rows:** 1,279 rows
- **Diagnostic Accuracy:** **45.50%** (vs 45.64% on full validation)
- **Diagnostic Macro F1:** **0.4445** (vs 0.4465 on full validation)
- **Diagnostic Macro Precision:** **0.4542**
- **Diagnostic Macro Recall:** **0.4471**

*Conclusion:* The marginal delta (-0.14% accuracy, -0.0020 F1) demonstrates that the minimal text overlap documented in Step 2 does not introduce material data leakage or artificially inflate validation performance.

---

## Observations & Next Steps
- The initial unweighted TF-IDF + Logistic Regression model achieved an accuracy of 45.64% and Macro F1 of 0.4465.
- Low recall on `likely_fake` (30.61%) indicated that unweighted classification favored majority categories.

---

## Baseline Hyperparameter Selection (Step 6)

> **Important:** All results reported in this section are validation results evaluated on `data/raw/valid.tsv`. The official test split (`data/raw/test.tsv`) remains strictly untouched.

### Search Space & Experimental Design
A controlled grid search evaluating **96 discrete configurations** was conducted using ONLY `data/raw/train.tsv` for fitting and `data/raw/valid.tsv` for model evaluation.

- **TF-IDF `ngram_range`:** `(1, 1)`, `(1, 2)`
- **TF-IDF `min_df`:** `1`, `2`, `3`
- **TF-IDF `max_features`:** `30,000`, `50,000`
- **Logistic Regression `C`:** `0.5`, `1.0`, `2.0`, `4.0`
- **Logistic Regression `class_weight`:** `None`, `"balanced"`
- **Fixed Parameters:** `lowercase=True`, `max_df=0.95`, `sublinear_tf=True`, `max_iter=1000`, `solver="lbfgs"`, `random_state=42`
- **Primary Selection Metric:** **Macro F1**

### Best Hyperparameter Configuration
- **TF-IDF:** `ngram_range=(1, 2)`, `min_df=1`, `max_features=50000`, `sublinear_tf=True`, `lowercase=True`
- **Logistic Regression:** `C=0.5`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`

### Initial vs. Tuned Baseline Comparison

| Metric | Initial Baseline (Step 5) | Tuned Baseline (Step 6) | Absolute Change |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 45.64% (0.4564) | **46.88%** (0.4688) | **+1.24%** (+0.0124) |
| **Macro Precision** | 0.4556 | **0.4677** | **+0.0121** |
| **Macro Recall** | 0.4486 | **0.4726** | **+0.0240** |
| **Macro F1 (Primary)** | 0.4465 | **0.4687** | **+0.0222** |
| **Weighted F1** | 0.4507 | **0.4672** | **+0.0165** |

### Per-Class Comparison (Before vs. After)

| Class | Initial Precision | Tuned Precision | Initial Recall | Tuned Recall | Initial F1 | Tuned F1 | Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **likely_fake** | 0.4462 | 0.4412 | 0.3061 | **0.4749** | 0.3631 | **0.4574** (+0.0943) | 379 |
| **likely_reliable** | 0.4753 | **0.4946** | 0.5262 | **0.5429** | 0.4994 | **0.5176** (+0.0182) | 420 |
| **misleading** | 0.4454 | **0.4675** | **0.5134** | 0.4000 | **0.4770** | 0.4311 (-0.0459) | 485 |

### Validation Confusion Matrix (Tuned Model)

*Learned Class Order:* `['likely_fake', 'likely_reliable', 'misleading']`  
*Format: Rows = Actual Ground Truth, Columns = Model Prediction*

| Actual \ Predicted | predicted: `likely_fake` | predicted: `likely_reliable` | predicted: `misleading` | Total |
| :--- | :--- | :--- | :--- | :--- |
| **actual: likely_fake** | **180** | 88 | 111 | 379 |
| **actual: likely_reliable** | 82 | **228** | 110 | 420 |
| **actual: misleading** | 146 | 145 | **194** | 485 |
| **Total Predicted** | 408 | 461 | 415 | 1,284 |

### Error Analysis & Honest Interpretation
1. **Resolution of Fake News Blind Spot:** In Step 5, unweighted Logistic Regression suffered from severe under-prediction of `likely_fake` (recall 30.61%). Applying `class_weight="balanced"` and regularization `C=0.5` dramatically boosted `likely_fake` recall from **30.61% to 47.49%** (+16.88 percentage points), raising its F1 score by **+0.0943** (0.3631 → 0.4574).
2. **Most Common Confusion Pair:** The most frequent error is actual `misleading` predicted as `likely_fake` (146 instances), closely followed by actual `misleading` predicted as `likely_reliable` (145 instances). Because `misleading` represents intermediate/nuanced claims (e.g., half-true, barely-true), linear bag-of-words features struggle to delineate the exact boundary between nuance and falsehood.
3. **Class Weighting Impact:** Models with `class_weight="balanced"` achieved a higher average Macro F1 (0.4381) than unweighted models (0.4330), demonstrating that addressing class imbalance is essential for this benchmark.
4. **Leakage-Sensitivity Diagnostic:**
   - Evaluated on 1,279 rows (excluding the 5 statements overlapping with train).
   - **Diagnostic Accuracy:** **46.68%** (vs 46.88% full validation).
   - **Diagnostic Macro F1:** **0.4665** (vs 0.4687 full validation).
   - Confirms that model selection was not driven by cross-split repetition.

### Decision
The tuned configuration (`models/baseline_best_validation.joblib`) achieved a meaningful Macro F1 improvement (+0.0222) and substantially better recall balance. It is selected as the candidate baseline for final official test evaluation in Step 7.

---

# Classical Model Comparison (Step 6.5)

> **Important:** All evaluations reported in this section were conducted strictly on `data/raw/valid.tsv` (1,284 rows) after fitting on `data/raw/train.tsv` (10,269 rows).  
> **The official test split (`data/raw/test.tsv`) remains completely untouched.**

## Purpose & Scope
Before proceeding to official test split evaluation, Step 6.5 explored whether alternative classical architectures—specifically Linear Support Vector Machines (LinearSVC) and sub-word character n-gram representations—could outperform the tuned Logistic Regression baseline on the validation set.

### Selection Rules & Thresholds
- **Primary Metric:** **Macro F1** across the 3 project classes (`likely_reliable`, `misleading`, `likely_fake`).
- **Meaningful Improvement Threshold:** $\ge +0.0020$ (+0.20 percentage points) over the Step 6 baseline reference (Macro F1 = 0.4687).
- **Tie-Breaking / Trade-Off Rule:** If no candidate exceeds the reference by $\ge +0.0020$, the reference Logistic Regression model is retained because it offers lower complexity, faster inference, and native class probability estimates via predict_proba required for downstream multi-model fusion.

---

## Candidate Model Families (23 Configurations)

1. **Model A (Reference):** Tuned Word TF-IDF + Logistic Regression (`ngram_range=(1, 2)`, `max_features=50000`, `C=0.5`, `class_weight='balanced'`).
2. **Model B:** Word TF-IDF + LinearSVC (6 configurations: $C \in \{0.5, 1.0, 2.0\}$, `class_weight` $\in \{\text{None}, \text{'balanced'}\}$).
3. **Model C:** Word + Sub-word Character TF-IDF (`char_wb`, n-grams 3–5) + Logistic Regression (8 configurations: `char_max_features` $\in \{30000, 50000\}$, $C \in \{0.5, 1.0\}$, `class_weight` $\in \{\text{'balanced'}, \text{None}\}$).
4. **Model D:** Word + Sub-word Character TF-IDF (`char_wb`, n-grams 3–5) + LinearSVC (8 configurations: `char_max_features` $\in \{30000, 50000\}$, $C \in \{0.5, 1.0\}$, `class_weight` $\in \{\text{None}, \text{'balanced'}\}$).

---

## Best Configuration per Model Family

| Family | Model Description | Best Hyperparameters | Accuracy | Macro F1 | Delta vs Step 6 | Native Probabilities | Fit Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Model A** | **Word TF-IDF + Logistic Regression (Step 6)** | **$C=0.5$, `balanced`, 50k word n-grams** | **46.88%** | **0.4687** | **+0.0000** | **Yes** | **1.33s** |
| Model B | Word TF-IDF + LinearSVC | $C=0.5$, unweighted, 50k word n-grams | 45.72% | 0.4496 | -0.0191 | No (`decision_function`) | 0.63s |
| Model C | Word + Char TF-IDF + Logistic Regression | $C=0.5$, `balanced`, 30k chars | 45.72% | 0.4575 | -0.0112 | Yes | 3.85s |
| Model D | Word + Char TF-IDF + LinearSVC | $C=0.5$, `balanced`, 50k chars | 43.93% | 0.4376 | -0.0311 | No (`decision_function`) | 2.27s |

---

## Complete Comparison Results (All 23 Evaluated Configurations)

| # | Group | Classifier | Word Features | Char Features | $C$ | Class Weight | Accuracy | Macro Prec | Macro Rec | Macro F1 | $\Delta$ Macro F1 | Fit Time | Native Proba |
| :-: | :-: | :--- | :--- | :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | **A** | **LogisticRegression** | **50,000 (1-2)** | **None** | **0.5** | **balanced** | **46.88%** | **0.4677** | **0.4726** | **0.4687** | **+0.0000** | **1.33s** | **Yes** |
| 2 | B | LinearSVC | 50,000 (1-2) | None | 0.5 | None | 45.72% | 0.4562 | 0.4510 | 0.4496 | -0.0191 | 0.63s | No |
| 3 | B | LinearSVC | 50,000 (1-2) | None | 0.5 | balanced | 44.78% | 0.4450 | 0.4454 | 0.4445 | -0.0242 | 0.67s | No |
| 4 | B | LinearSVC | 50,000 (1-2) | None | 1.0 | None | 43.93% | 0.4364 | 0.4335 | 0.4322 | -0.0365 | 0.72s | No |
| 5 | B | LinearSVC | 50,000 (1-2) | None | 1.0 | balanced | 43.85% | 0.4357 | 0.4357 | 0.4350 | -0.0337 | 0.73s | No |
| 6 | B | LinearSVC | 50,000 (1-2) | None | 2.0 | None | 43.30% | 0.4304 | 0.4282 | 0.4277 | -0.0410 | 0.83s | No |
| 7 | B | LinearSVC | 50,000 (1-2) | None | 2.0 | balanced | 43.07% | 0.4287 | 0.4279 | 0.4277 | -0.0410 | 0.60s | No |
| 8 | C | LogisticRegression | 50,000 (1-2) | 30,000 (3-5) | 0.5 | balanced | 45.72% | 0.4564 | 0.4605 | 0.4575 | -0.0112 | 3.85s | Yes |
| 9 | C | LogisticRegression | 50,000 (1-2) | 30,000 (3-5) | 0.5 | None | 45.95% | 0.4628 | 0.4545 | 0.4538 | -0.0149 | 2.20s | Yes |
| 10 | C | LogisticRegression | 50,000 (1-2) | 30,000 (3-5) | 1.0 | balanced | 44.31% | 0.4422 | 0.4446 | 0.4431 | -0.0256 | 4.84s | Yes |
| 11 | C | LogisticRegression | 50,000 (1-2) | 30,000 (3-5) | 1.0 | None | 44.78% | 0.4490 | 0.4440 | 0.4438 | -0.0249 | 2.86s | Yes |
| 12 | C | LogisticRegression | 50,000 (1-2) | 50,000 (3-5) | 0.5 | balanced | 45.40% | 0.4534 | 0.4573 | 0.4546 | -0.0141 | 3.89s | Yes |
| 13 | C | LogisticRegression | 50,000 (1-2) | 50,000 (3-5) | 0.5 | None | 45.64% | 0.4593 | 0.4511 | 0.4500 | -0.0187 | 2.45s | Yes |
| 14 | C | LogisticRegression | 50,000 (1-2) | 50,000 (3-5) | 1.0 | balanced | 44.16% | 0.4411 | 0.4427 | 0.4417 | -0.0270 | 4.71s | Yes |
| 15 | C | LogisticRegression | 50,000 (1-2) | 50,000 (3-5) | 1.0 | None | 44.94% | 0.4515 | 0.4450 | 0.4452 | -0.0235 | 3.44s | Yes |
| 16 | D | LinearSVC | 50,000 (1-2) | 30,000 (3-5) | 0.5 | None | 43.22% | 0.4305 | 0.4283 | 0.4274 | -0.0413 | 2.33s | No |
| 17 | D | LinearSVC | 50,000 (1-2) | 30,000 (3-5) | 0.5 | balanced | 43.61% | 0.4353 | 0.4350 | 0.4346 | -0.0341 | 2.27s | No |
| 18 | D | LinearSVC | 50,000 (1-2) | 30,000 (3-5) | 1.0 | None | 44.00% | 0.4383 | 0.4367 | 0.4363 | -0.0324 | 2.73s | No |
| 19 | D | LinearSVC | 50,000 (1-2) | 30,000 (3-5) | 1.0 | balanced | 43.38% | 0.4314 | 0.4318 | 0.4312 | -0.0375 | 2.71s | No |
| 20 | D | LinearSVC | 50,000 (1-2) | 50,000 (3-5) | 0.5 | None | 43.38% | 0.4325 | 0.4301 | 0.4294 | -0.0393 | 2.27s | No |
| 21 | D | LinearSVC | 50,000 (1-2) | 50,000 (3-5) | 0.5 | balanced | 43.93% | 0.4386 | 0.4381 | 0.4376 | -0.0311 | 2.27s | No |
| 22 | D | LinearSVC | 50,000 (1-2) | 50,000 (3-5) | 1.0 | None | 44.00% | 0.4375 | 0.4371 | 0.4363 | -0.0324 | 2.78s | No |
| 23 | D | LinearSVC | 50,000 (1-2) | 50,000 (3-5) | 1.0 | balanced | 43.61% | 0.4334 | 0.4339 | 0.4332 | -0.0355 | 2.77s | No |

---

## Detailed Analysis of Model Selection

### 1. Performance of LinearSVC vs. Logistic Regression
- Across all configurations, LinearSVC systematically underperformed Logistic Regression.
- The best Word TF-IDF LinearSVC reached **Macro F1 = 0.4496** ($C=0.5$), trailing the Logistic Regression reference by **0.0191**.
- As regularization relaxed ($C=1.0 \rightarrow 2.0$), LinearSVC performance degraded further (down to 0.4277).
- *Observation:* LinearSVC produced lower validation Macro F1 in the tested configurations. One possible explanation is substantial lexical overlap among the mapped classes, but this experiment does not establish causality.

### 2. Impact of Character n-grams
- Adding sub-word character n-grams (`char_wb`, n-grams 3–5) did NOT improve validation performance for either classifier.
- For Logistic Regression, adding 30,000 character features reduced Macro F1 from **0.4687 down to 0.4575** (-0.0112), while increasing training time from **1.33s to 3.85s**.
- Expanding character features to 50,000 further degraded Macro F1 to **0.4546** (-0.0141).
- *Observation:* Adding character n-gram features increased feature dimensionality but did not improve validation performance in the tested configurations.

### 3. Class Probability Estimates & Pipeline Simplicity
- Logistic Regression provides native class probability estimates via predict_proba (via softmax). Note that class probability estimates returned by Logistic Regression are not assumed to be calibrated unless calibration is separately evaluated.
- LinearSVC provides only signed margin distances (`decision_function`). Converting margin distances to probability estimates would require additional post-hoc steps (e.g., Platt scaling or isotonic regression), whereas Logistic Regression provides native class probability estimates via predict_proba directly.
- Native class probability estimates via predict_proba are useful for downstream multi-model fusion where classical probability estimates will be combined with BERT and LLM outputs.

---

## Final Selected Classical Model

The **Tuned Logistic Regression** model ($C=0.5$, `class_weight='balanced'`, word n-grams 1–2, 50,000 features) is decisively selected and saved as:
- **Model Artifact:** `models/classical_best_validation.joblib`
- **Metadata:** `models/classical_best_validation_metrics.json`
- **Confusion Matrix:** `models/classical_best_validation_confusion_matrix.csv`

### Final Selected Model Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **likely_fake** | 0.4412 | 0.4749 | **0.4574** | 379 |
| **likely_reliable** | 0.4946 | 0.5429 | **0.5176** | 420 |
| **misleading** | 0.4675 | 0.4000 | **0.4311** | 485 |
| **Macro Average** | **0.4677** | **0.4726** | **0.4687** | **1,284** |
| **Accuracy** | — | — | **46.88%** | **1,284** |

### Confusion Matrix

| Actual \ Predicted | predicted: `likely_fake` | predicted: `likely_reliable` | predicted: `misleading` | Total |
| :--- | :--- | :--- | :--- | :--- |
| **actual: likely_fake** | **180** | 88 | 111 | 379 |
| **actual: likely_reliable** | 82 | **228** | 110 | 420 |
| **actual: misleading** | 146 | 145 | **194** | 485 |
| **Total Predicted** | 408 | 461 | 415 | 1,284 |

### Verification of Frozen Artifacts
- `models/baseline_initial.joblib`: Preserved (Step 5 initial baseline, Macro F1 = 0.4465).
- `models/baseline_best_validation.joblib`: Preserved (Step 6 tuned baseline, Macro F1 = 0.4687).
- `models/classical_best_validation.joblib`: Verified and frozen (Step 6.5 selected classical model, Macro F1 = 0.4687).

---

# Final Classical Model Test Evaluation (Step 7)

> **Important:** This section documents the one-time, final out-of-sample evaluation of the frozen classical model on the official LIAR benchmark test split (`data/raw/test.tsv`).  
> **Zero model fitting was performed.** The evaluation script loaded `models/classical_best_validation.joblib` and ran pure inference (`predict` and `predict_proba`).

---

## 1. Frozen Model Specification

- **Architecture:** Word TF-IDF + Logistic Regression
- **Frozen Model Artifact:** `models/classical_best_validation.joblib`
- **Model Selection Protocol:** Selected strictly using validation split performance (`data/raw/valid.tsv`) during Step 6 and Step 6.5. No test split data was accessed during model selection or hyperparameter tuning.
- **Exact Configuration:**
  - **TF-IDF Vectorizer:** `lowercase=True`, `ngram_range=(1, 2)`, `min_df=1`, `max_df=0.95`, `max_features=50,000`, `sublinear_tf=True`
  - **Logistic Regression Classifier:** `C=0.5`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`

---

## 2. Test Dataset Profile

- **Source File:** `data/raw/test.tsv`
- **Total Evaluated Rows:** **1,283** (100% of test records evaluated, zero rows omitted or dropped)
- **Mapped Class Distribution:**
  - `misleading`: **481** (37.49%)
  - `likely_reliable`: **460** (35.85%)
  - `likely_fake`: **342** (26.66%)

---

## 3. Official Test Performance Metrics

| Metric | Official Test Score | Validation Reference (Step 6.5) | Delta vs. Validation |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **43.73%** (0.4373) | 46.88% (0.4688) | -3.15% (-0.0315) |
| **Macro Precision** | **0.4339** | 0.4677 | -0.0338 |
| **Macro Recall** | **0.4381** | 0.4726 | -0.0345 |
| **Macro F1** | **0.4345** | 0.4687 | -0.0342 |
| **Weighted F1** | **0.4362** | 0.4672 | -0.0310 |

### Per-Class Test Performance Breakdown

| Class | Precision | Recall | F1-Score | Support (Actual Count) |
| :--- | :---: | :---: | :---: | :---: |
| **likely_fake** | 0.3880 | 0.4357 | **0.4105** | 342 |
| **likely_reliable** | 0.4735 | 0.5043 | **0.4884** | 460 |
| **misleading** | 0.4401 | 0.3742 | **0.4045** | 481 |
| **Macro Average** | **0.4339** | **0.4381** | **0.4345** | **1,283** |
| **Weighted Average** | **0.4382** | **0.4373** | **0.4362** | **1,283** |

---

## 4. Official Test Confusion Matrix

*Learned Class Order:* `['likely_fake', 'likely_reliable', 'misleading']`  
*Format: Rows = Actual Ground Truth Label, Columns = Model Prediction*  
*Source File:* `models/classical_test_confusion_matrix.csv`

| Actual \ Predicted | predicted: `likely_fake` | predicted: `likely_reliable` | predicted: `misleading` | Total |
| :--- | :---: | :---: | :---: | :---: |
| **actual: likely_fake** | **149** | 89 | 104 | 342 |
| **actual: likely_reliable** | 103 | **232** | 125 | 460 |
| **actual: misleading** | 132 | 169 | **180** | 481 |
| **Total Predicted** | 384 | 490 | 409 | 1,283 |

---

## 5. Validation vs. Test Comparative Analysis

### Metric Deltas
- **Accuracy Difference:** $0.4373 - 0.4688 = \mathbf{-0.0315}$ (-3.15 percentage points)
- **Macro F1 Difference:** $0.4345 - 0.4687 = \mathbf{-0.0342}$ (-3.42 percentage points)
- **Per-Class F1 Deltas:**
  - `likely_reliable`: $0.5176 \rightarrow 0.4884$ ($\Delta = -0.0292$)
  - `misleading`: $0.4311 \rightarrow 0.4045$ ($\Delta = -0.0266$)
  - `likely_fake`: $0.4574 \rightarrow 0.4105$ ($\Delta = -0.0469$)

### Conservative Interpretation
- Test Macro F1 was modestly lower than validation (-0.0342), which is expected when evaluating on an untouched, chronologically distinct benchmark split.
- The relative ordering of per-class performance remained consistent: `likely_reliable` achieved the strongest F1 score on both splits (0.5176 on validation, 0.4884 on test), while `misleading` and `likely_fake` presented similar classification difficulty.
- Class-weight balancing succeeded in maintaining substantial recall for `likely_fake` on the test set (**43.57%**), preventing the severe false-negative collapse observed in the unweighted initial baseline (where recall was only 30.61%).

---

## 6. Overlap-Excluded Sensitivity Diagnostic (Secondary)

Step 2 identified 4 statements in the test set that appear verbatim in the training set. Official metrics above evaluate all 1,283 rows. As a secondary sensitivity check, metrics were computed on the 1,279 non-overlapping test rows:

- **Excluded Overlapping Statements:** 4 rows
- **Evaluated Diagnostic Rows:** 1,279 rows
- **Diagnostic Accuracy:** **43.63%** (vs 43.73% official test)
- **Diagnostic Macro Precision:** **0.4328** (vs 0.4339 official test)
- **Diagnostic Macro Recall:** **0.4369** (vs 0.4381 official test)
- **Diagnostic Macro F1:** **0.4333** (vs 0.4345 official test)

*Conclusion:* The negligible delta ($\Delta \text{Acc} = -0.10\%$, $\Delta \text{F1} = -0.0012$) confirms that the tiny cross-split overlap did not artificially inflate the official test score.

---

## 7. Descriptive Probability Output Diagnostic

Analyzing the maximum predicted class probability estimate for each test prediction:

| Prediction Group | Sample Count | Mean Max Probability | Median Max Probability | Min Max Probability | Max Max Probability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correct Predictions** | 561 | **0.4511** | 0.4373 | 0.3407 | 0.7815 |
| **Incorrect Predictions** | 722 | **0.4325** | 0.4169 | 0.3369 | 0.7438 |

*Descriptive Note:* Both correct and incorrect predictions exhibit mean maximum probability estimates in the 0.43–0.45 range (against a random-guess baseline of 0.33), highlighting the uncertainty inherent in linear text-only classification of short claims. Class probability estimates returned by Logistic Regression are not assumed to be calibrated unless calibration is separately evaluated.

---

## 8. Fundamental Model Limitations & Motivation for Subsequent Layers

1. **Statistical Text-Pattern Classification vs. Evidence Verification:**
   "The classical model is a statistical text-pattern classifier. It does not verify claims using external factual evidence." It learns associations between lexical tokens and truth labels in the training corpus. When a fabricated statement uses authoritative, professional language, bag-of-words models frequently misclassify it as reliable.
2. **Probability Calibration Limits:**
   "Class probability estimates returned by Logistic Regression are not assumed to be calibrated unless calibration is separately evaluated."
3. **Motivation for Project Architecture:**
   These limitations directly motivate the multi-stage, hybrid architecture of the overall project:
   - **Step 8:** Interpretable stylometric and red-flag feature analysis (hedging, sentiment extremity, clickbait syntax).
   - **Subsequent Steps:** LLM claim extraction, automated external evidence retrieval via search/corpus lookups, structured LLM reasoning, Bayesian/multi-model fusion, and principled abstention (`unverifiable`).

---

## 9. Final Classical Baseline Freeze

With Step 7 complete:
- The classical baseline model architecture and its test evaluation results are **permanently frozen**.
- No hyperparameter tuning or architectural adjustments may be performed based on test performance.
- All test artifacts (`classical_test_metrics.json`, `classical_test_confusion_matrix.csv`, `classical_test_predictions.csv`) are finalized and committed to the project record.



