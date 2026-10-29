# V2 Dataset Harmonization & Corpus Engineering Report

**Project:** Fake News Detection Using LLMs (`P_105`)  
**Student:** Bhagya Varshney  
**Mapping Version:** `v2_label_mapping_v1`  
**Generated:** 2026-10-18T09:00:46.222405+00:00  
**Status:** Dataset Engineering Complete — Zero Models Trained  

---

## 1. Executive Summary

In Step **V2.4 / V2.4.1**, we audited, downloaded, normalized, deduplicated, and harmonized broader fact-checking datasets to design a scientifically defensible training corpus for future classifier experimentation (Step V2.5). Strict methodological principles were enforced:
- **Zero Model Training:** No classifiers were retrained or evaluated in this step.
- **Zero LLM Relabeling:** Groq was used 0 times. Ground truth annotations remain purely author/publisher-provided.
- **Strict Held-Out Protection:** The frozen LIAR test split (1,283 rows) and AVeriTeC development split (500 rows) were quarantined with cryptographic hash checks. Zero test samples entered the training pool.
- **Reconciled LIAR Parsing:** LIAR TSV parsing strictly adheres to `quoting=csv.QUOTE_NONE`, reconciling the historical Step 2 counts (10,269 train, 1,284 valid, 1,283 test) and eliminating quote-merging truncation.
- **Licensing Compliance:** MultiFC was designated as `research candidate — licensing unresolved` and quarantined from the canonical training pool. FEVER was designated as `auxiliary` due to its Wikipedia-rewrite distribution shift.

---

## 2. Ingestion & Preprocessing Statistics

| Source Dataset | Official Split | Raw Samples | Harmonized Status | Protected Test? | License |
|---|---|---|---|---|---|
| **LIAR** | Train (`train.tsv`) | 10,269 | Eligible | No | Academic Research |
| **LIAR** | Valid (`valid.tsv`) | 1,284 | Preserved Independent Validation | No | Academic Research |
| **LIAR** | Test (`test.tsv`) | 1,283 | Protected Benchmark | **YES** | Academic Research |
| **AVeriTeC** | Train (`train.json`) | 3,068 | Eligible | No | CC BY-NC 4.0 |
| **AVeriTeC** | Dev (`dev.json`) | 500 | Protected Benchmark | **YES** | CC BY-NC 4.0 |
| **MultiFC** | Train (`train.tsv`) | 27,871 | Quarantined (Licensing) | No | Unresolved / Unknown |
| **MultiFC** | Dev (`dev.tsv`) | 3,484 | Quarantined (Licensing) | No | Unresolved / Unknown |
| **MultiFC** | Test (`test.tsv`) | 3,487 | Quarantined (Dummy Labels) | **YES** | Unresolved / Unknown |
| **FEVER** | Shared Dev (`shared_task_dev.jsonl`) | 19,998 | Auxiliary Stance Corpus | **YES** | CC BY-SA 3.0 |
| **Total Ingested** | All Sources | **71244** | — | — | — |

---

## 3. Label Harmonization Mapping Tables

### A. LIAR Mapping (Preserving V1 Baseline)
| Original LIAR Label | Harmonized V2 Class | Confidence | Mapping Status | Mapping Rule |
|---|---|---|---|---|
| `true` | `likely_reliable` | 1.00 | Accepted | `liar_v1_true_to_reliable` |
| `mostly-true` | `likely_reliable` | 0.95 | Accepted | `liar_v1_mostly_true_to_reliable` |
| `half-true` | `misleading` | 0.95 | Accepted | `liar_v1_half_true_to_misleading` |
| `barely-true` | `misleading` | 0.90 | Accepted | `liar_v1_barely_true_to_misleading` |
| `false` | `likely_fake` | 1.00 | Accepted | `liar_v1_false_to_fake` |
| `pants-fire` | `likely_fake` | 1.00 | Accepted | `liar_v1_pants_fire_to_fake` |

### B. AVeriTeC Mapping (Audited from NeurIPS 2023)
| Original AVeriTeC Label | Harmonized V2 Class | Confidence | Mapping Status | Mapping Rationale |
|---|---|---|---|---|
| `Supported` | `likely_reliable` | 1.00 | Accepted | Direct evidentiary proof from authoritative web sources. |
| `Refuted` | `likely_fake` | 1.00 | Accepted | Direct evidentiary contradiction from authoritative web sources. |
| `Not Enough Evidence` | `unverifiable` | 1.00 | Accepted | Exhaustive web retrieval found insufficient authoritative reporting. |
| `Conflicting Evidence/Cherrypicking` | `misleading` | 0.85 | Accepted | Selective context omission or empirical standoff across outlets. |

---

## 4. Canonical Eligible Training Pool Composition

The canonical training pool (`data/v2_processed/combined_eligible_train_pool.jsonl`) comprises **12192** verified claims constructed strictly from `LIAR train` and `AVeriTeC train` (preserving `LIAR validation` as an independent validation set):
- **LIAR Train:** 9240 samples (75.79%)
- **AVeriTeC Train:** 2952 samples (24.21%)

### Target Class Distribution in Canonical Training Pool
| Target Class | Sample Count | Percentage of Training Pool |
|---|---|---|
| `likely_reliable` | 4062 | 33.32% |
| `misleading` | 3589 | 29.44% |
| `likely_fake` | 4275 | 35.06% |
| `unverifiable` | 266 | 2.18% |

> [!IMPORTANT]
> **Zero Rebalancing in V2.4:** No SMOTE, oversampling, undersampling, or class weighting was applied. The class distribution above represents the true empirical density of the audited corpus. Class rebalancing and focal loss experimentation are explicitly deferred to V2.5.

---

## 5. Deduplication and Cross-Split Contamination Audit

- **Protected Test Hashes Quarantined:** 24733 distinct normalized hashes.
- **Cross-Split Contamination Eliminated:** 1028 candidate training samples were dropped because their normalized text matched an official test/dev benchmark claim.
- **Exact Duplicates Dropped:** 116 redundant training items removed.

### Cross-Dataset Claim Overlap Analysis
| Dataset Comparison | Exact Normalized Text Overlap |
|---|---|
| `liar_vs_averitec` | 0 |
| `liar_vs_multifc` | 12325 |
| `liar_vs_fever` | 0 |
| `averitec_vs_multifc` | 4 |
| `averitec_vs_fever` | 0 |
| `multifc_vs_fever` | 0 |

---

## 6. V2.5 Validation Policy: Stratified 5-Fold Cross-Validation

To prevent validation folds from being accidentally dominated by one dataset or skewed by class imbalance, cross-validation is stratified by compound key `source_dataset + harmonized_label`.

| Fold | Total Validation Samples | LIAR Samples | LIAR % | AVeriTeC Samples | AVeriTeC % | Likely Fake | Likely Reliable | Misleading | Unverifiable |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 2439 | 1848 | 75.77% | 591 | 24.23% | 855 | 813 | 718 | 53 |
| Fold 2 | 2439 | 1848 | 75.77% | 591 | 24.23% | 856 | 812 | 717 | 54 |
| Fold 3 | 2438 | 1848 | 75.8% | 590 | 24.2% | 855 | 812 | 718 | 53 |
| Fold 4 | 2438 | 1848 | 75.8% | 590 | 24.2% | 855 | 812 | 718 | 53 |
| Fold 5 | 2438 | 1848 | 75.8% | 590 | 24.2% | 854 | 813 | 718 | 53 |

---

## 7. Recommended V2.5 Modeling & Evaluation Protocol

1. **Model Selection:**
   - Use the stratified 5-fold cross-validation scheme above over `combined_eligible_train_pool.jsonl` (12,192 claims).
   - In each fold, compute overall Macro F1, LIAR sub-split Macro F1, and AVeriTeC sub-split Macro F1.
2. **Independent Validation Benchmark:**
   - Use the preserved `LIAR validation` split (1,284 claims) as an out-of-fold historical validation check.
3. **Final Held-Out Benchmarks (Zero Hyperparameter Tuning):**
   - **LIAR Frozen Test (`data/raw/test.tsv` — 1,283 claims):** Evaluates historical political baseline retention.
   - **AVeriTeC Protected Dev (`data/v2_raw/averitec/dev.json` — 500 claims):** Evaluates modern web-retrieval generalization.

---
*Report generated automatically by `scripts/prepare_v2_datasets.py`.*
