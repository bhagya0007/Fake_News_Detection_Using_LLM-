# V2 Model Training and Multi-Domain Evaluation Plan

**Project:** Fake News Detection Using LLMs (`P_105`)  
**Student:** Bhagya Varshney  
**Phase:** V2.4 Dataset Engineering & V2.5 Preparation  
**Status:** Canonical Evaluation Protocol Specification  

---

## 1. Overview and Core Epistemic Principles

The primary limitation of traditional machine learning benchmarks in fake news detection is **dataset overfitting and spurious correlation exploitation** (e.g., learning publisher writing styles or domain-specific political jargon rather than generalizing factual credibility indicators).

In Version 2, Project P_105 expands from a single-dataset classical baseline (LIAR: 43.73% Accuracy, 0.4345 Macro F1) into a multi-domain, evidence-grounded framework. To prevent larger or idiosyncratic datasets from masking catastrophic domain shifts, **future evaluation in V2.5 must NEVER report solely a single aggregated accuracy metric**.

Instead, evaluation will proceed under a **stratified, multi-benchmark protocol** that isolates in-distribution performance from out-of-domain generalization.

---

## 2. Train / Test Split Protection & Model Selection Policy

### A. The Frozen LIAR Benchmark Invariant
- The official LIAR test split (`data/raw/test.tsv` — **1,283 rows**, SHA-256: `ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f`) has already been used to evaluate the V1 frozen baseline (`models/classical_best_validation.joblib`, SHA-256: `d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92`).
- **Strict Rule:** The LIAR test set **MUST NEVER be used to tune hyperparameters, select decision thresholds, or optimize feature extractors** in V2.5.
- Doing so would invalidate historical baseline comparisons due to benchmark leakage.
- In V2.5, the LIAR test set will serve strictly as a **frozen secondary benchmark** to measure whether cross-domain training improved or degraded historical political claim performance.

### B. V2 Model Selection Strategy: Dataset-Aware 5-Fold Cross-Validation
- All model selection, hyperparameter sweeps (e.g., regularization $C$, n-gram ranges, probability calibration, GBDT depth), and feature engineering in V2.5 will be validated using **Stratified 5-Fold Cross-Validation on the canonical training pool** (`data/v2_processed/combined_eligible_train_pool.jsonl`, **12,192 claims**).
- Folds are stratified using the compound key `source_dataset + harmonized_label` to ensure exact, balanced representation of both LIAR (~75.8%) and AVeriTeC (~24.2%) across every fold.
- `LIAR validation` (`data/raw/valid.tsv` — **1,284 claims**) is preserved separately as an independent out-of-fold historical validation check.

#### Dataset-Source Balance Across Proposed Validation Folds
| Fold | Total Validation Samples | LIAR Samples | LIAR % | AVeriTeC Samples | AVeriTeC % | Likely Fake | Likely Reliable | Misleading | Unverifiable |
|---|---|---|---|---|---|---|---|---|---|
| **Fold 1** | 2,439 | 1,848 | 75.77% | 591 | 24.23% | 855 | 813 | 718 | 53 |
| **Fold 2** | 2,439 | 1,848 | 75.77% | 591 | 24.23% | 856 | 812 | 717 | 54 |
| **Fold 3** | 2,438 | 1,848 | 75.80% | 590 | 24.20% | 855 | 812 | 718 | 53 |
| **Fold 4** | 2,438 | 1,848 | 75.80% | 590 | 24.20% | 855 | 812 | 718 | 53 |
| **Fold 5** | 2,438 | 1,848 | 75.80% | 590 | 24.20% | 854 | 813 | 718 | 53 |

---

## 3. Dedicated Evaluation Benchmarks

In V2.5, any trained candidate model will be evaluated across **two mandatory distinct held-out test splits**:

```
                       Trained V2 Model
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
[Benchmark 1: LIAR Frozen Test]       [Benchmark 2: AVeriTeC Held-Out Dev]
  • 1,283 claims                        • 500 claims
  • US Political Statements             • Global Multi-Domain Facts (50 orgs)
  • Short snippets (no web QA)          • Web Evidence & QA Justifications
  • Historical Baseline Comparison      • Modern Real-World Fact-Check Benchmark
```

### Benchmark 1: LIAR Frozen Test Split
- **Source:** `data/raw/test.tsv` (1,283 samples)
- **Role:** Historical baseline comparative evaluation.
- **Target Metric Goal:** Compare against V1 baseline (Accuracy: 43.73%, Macro F1: 0.4345).
- **Labels:** 460 `likely_reliable`, 481 `misleading`, 342 `likely_fake`, 0 `unverifiable`.

### Benchmark 2: AVeriTeC Held-Out Benchmark
- **Source:** `data/v2_raw/averitec/dev.json` (500 samples)
- **Role:** Out-of-domain generalization and web-retrieval benchmark.
- **Target Metric Goal:** Evaluate multi-class classification across international public health, science, economics, and viral misinformation.
- **Labels:** 122 `likely_reliable`, 38 `misleading`, 305 `likely_fake`, 35 `unverifiable`.

### Benchmark 3: MultiFC (Conditional Research Benchmark)
- **Source:** `data/v2_raw/multifc/dev.tsv` / `test.tsv`
- **Role:** Only evaluated if licensing terms are formally resolved.
- **Condition:** Must report per-checker accuracy across the 26 fact-checking sites to expose publisher-specific label biases.

---

## 4. Required Reporting Metrics per Evaluation Run

For every model trained in V2.5, the evaluation harness must generate the following metrics **both overall and broken down individually per benchmark**:

1. **Overall Accuracy:** $\frac{\text{Correct Predictions}}{\text{Total Samples}}$
2. **Macro-Averaged F1 Score:** $\frac{1}{K} \sum_{k=1}^K F1_k$ (unweighted mean across classes; primary optimization target to prevent majority-class bias).
3. **Per-Class Metrics:**
   - Precision, Recall, and F1 for `likely_reliable`
   - Precision, Recall, and F1 for `misleading`
   - Precision, Recall, and F1 for `likely_fake`
   - Precision, Recall, and F1 for `unverifiable`
4. **Normalized Confusion Matrix ($4 \times 4$):**
   - Displays true label vs. predicted label to diagnose asymmetric error patterns (e.g., whether `likely_fake` claims are mistakenly classified as `likely_reliable` or safely degraded into `misleading`/`unverifiable`).
5. **Epistemic Abstention Quality:**
   - Fraction of claims with insufficient web evidence or low classifier confidence that successfully map to `unverifiable`.

---

## 5. Candidate Model Architectures for V2.5 Experimentation

The V2.4 training pool (`combined_eligible_train_pool.jsonl`, 13,319 verified claims) will support the following experimental trajectory in V2.5:

1. **Baseline M0 (Frozen V1 Reference):**
   - TF-IDF + Logistic Regression trained on LIAR train only.
2. **Model M1 (Multi-Domain Linear Baseline):**
   - Expanded sublinear TF-IDF (word + char n-grams) + Calibrated Logistic Regression trained on `combined_eligible_train_pool.jsonl`.
3. **Model M2 (Gradient Boosted Decision Trees):**
   - LightGBM / GBDT with semantic sentence embeddings (`all-MiniLM-L6-v2`) and metadata features (speaker historical credit, domain).
4. **Model M3 (Compact Neural / Transformer Classifier):**
   - Fine-tuned lightweight encoder (e.g., DeBERTa-v3-small or DistilBERT) evaluated under strict offline compute constraints.

---

## 6. Acceptance Criteria for Model Selection in V2.5

To replace Model M0 as the active classical/ML component in the hybrid pipeline:
1. Candidate model must demonstrate **statistically significant improvement in Macro F1 on AVeriTeC benchmark** without catastrophic regression ($>2\%$ drop) on LIAR frozen test benchmark.
2. Model must emit well-calibrated confidence probabilities ($P \in [0, 1]$) compatible with the hybrid fusion engine's threshold gates.
3. Inference latency must remain under $50\text{ ms}$ per claim to preserve the $<1.5\text{ s}$ total pipeline response budget.
