# Comprehensive Final System Evaluation Report

**Project:** Fake News Detection Using LLMs  
**Project ID:** P_105  
**Student:** Bhagya Varshney  
**System Classification:** Functional Academic Prototype (Not a Global Fact-Checker)  
**Evaluation Date:** October 2026  

---

## Executive Summary

This evaluation report documents the empirical and functional evaluation of the complete hybrid fake news detection system across three primary evaluation tracks, three subsystem functional benchmarks, and rigorous latency and resource profiling.

### Key Multi-Track Evaluation Summary
1. **Track A (Controlled Hybrid Functional Benchmark):** Overall Accuracy **100.0%**, Macro F1 **1.0000**, Abstention Accuracy **100.0%** across 44 balanced controlled scenarios. Unsafe ungrounded decisions: **0**.
2. **Track B (Production-Mode Real-World Evaluation):** Evaluated against an empty production knowledge base (`evidence_kb.jsonl`, 0 records). Abstention rate was **100.0%** (20/20 statements abstained as `unverifiable` with `support_score = 0.0`), validating that without verified local evidence, the system refuses to emit hallucinated fact-checks.
3. **Track C (Frozen Classical ML Baseline):** Evaluated on the official frozen LIAR test split (`models/classical_test_predictions.csv`, 1,283 samples). Test Accuracy **43.73%**, Test Macro F1 **0.4345**.
4. **Safety & Policy Invariants:** **10 / 10** formal fusion invariants passed, confirming strict evidence gating, synthetic fixture quarantine, and stylometric non-conviction guarantees.

---

## 1. Prototype Classification & Evaluation Framing

- **Academic Prototype Status:** This system is engineered as an educational and research prototype demonstrating hybrid multi-signal fusion, not an open-domain web fact-checking service.
- **Separation of Metrics:** Synthetic benchmark performance (Track A) is strictly quarantined from empirical LIAR test benchmark metrics (Track C). Under no circumstances are synthetic accuracy scores presented as general fake-news detection accuracy.
- **Decision-Support Score Semantics:** `support_score` is an **uncalibrated heuristic decision-support score** (0.00 to 1.00) measuring evidentiary corroboration and multi-signal alignment. It is NOT an objective or calibrated probability of factual truth.
- **Zero-Evidence Safe Abstention:** In production mode, the local knowledge base contains 0 ingested records (`data/evidence/evidence_kb.jsonl`, 0 bytes). The system is designed to abstain safely rather than extrapolate or guess.

---

## 2. Track A: Controlled Hybrid Functional Benchmark Results

The controlled benchmark evaluated 44 carefully curated multi-signal cases across 15 distinct behavioral categories with balanced ground-truth labels (11 Likely Reliable, 11 Likely Fake, 11 Misleading, 11 Unverifiable).

### Overall Metrics
- **Total Evaluated Cases:** 44
- **Overall Accuracy:** 100.00% (44/44)
- **Macro F1 Score:** 1.0000
- **Abstention Accuracy:** 100.00%
- **Unsafe Decision Count:** 0 (Zero ungrounded determinations when evidence was absent)
- **Evidence Grounding Violations:** 0
- **Synthetic Quarantine Violations:** 0

### Per-Class Performance Breakdown
| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `likely_reliable` | 1.0000 | 1.0000 | 1.0000 | 11 |
| `likely_fake` | 1.0000 | 1.0000 | 1.0000 | 11 |
| `misleading` | 1.0000 | 1.0000 | 1.0000 | 11 |
| `unverifiable` | 1.0000 | 1.0000 | 1.0000 | 11 |

### Category-Specific Functional Behavior
| Behavioral Category | Evaluated Cases | Correct Decisions | Accuracy |
| :--- | :---: | :---: | :---: |
| `supported_claim` | 4 | 4 | 100.0% |
| `ml_llm_disagreement` | 4 | 4 | 100.0% |
| `red_flags_supporting_evidence` | 2 | 2 | 100.0% |
| `attribution_sensitive` | 3 | 3 | 100.0% |
| `numeric_statistical_claim` | 4 | 4 | 100.0% |
| `multi_claim_statement` | 4 | 4 | 100.0% |
| `contradicted_claim` | 4 | 4 | 100.0% |
| `red_flags_contradiction` | 2 | 2 | 100.0% |
| `partial_misleading_claim` | 5 | 5 | 100.0% |
| `red_flags_misleading` | 1 | 1 | 100.0% |
| `no_evidence` | 2 | 2 | 100.0% |
| `non_checkable_opinion` | 2 | 2 | 100.0% |
| `conflicting_evidence` | 2 | 2 | 100.0% |
| `synthetic_quarantine_prod` | 2 | 2 | 100.0% |
| `llm_unavailable_fallback` | 2 | 2 | 100.0% |
| `retrieval_rejection_low_sim` | 1 | 1 | 100.0% |

---

## 3. Track B: Production-Mode Abstention & Real-World Coverage

Track B evaluated 20 diverse factual and non-factual statements spanning 10+ real-world domains (politics, healthcare, macroeconomics, climate, artificial intelligence, sports, astrophysics, transit, sensational hoaxes, and subjective opinion) against the empty production repository.

### Empirical Behavior Summary
- **Total Statements Evaluated:** 20
- **Abstention Count:** 20 / 20
- **Abstention Rate:** 100.0%
- **Verdict Distribution:** 100.0% `unverifiable`
- **Mean Support Score:** 0.00 / 1.00
- **LLM Short-Circuit Rate:** 100.0% (Zero LLM reasoning tokens consumed)
- **Average Red Flags Detected:** 0.50 per statement (Max: 3)

### Classical ML Predictions on Production Inputs (Without Evidence)
When evaluated through the frozen TF-IDF + Logistic Regression model alone:
- `likely_fake`: 7 cases (35.0%)
- `misleading`: 6 cases (30.0%)
- `likely_reliable`: 7 cases (35.0%)

> [!IMPORTANT]
> Despite the classical ML model producing strong text-pattern predictions for these 20 statements, the hybrid fusion engine refused to endorse any of them. Because zero provenance-bearing evidence was present in `data/evidence/evidence_kb.jsonl`, the evidence gate intervened and forced safe abstention.

---

## 4. Track C: Frozen Classical Model Error Analysis (LIAR Test Split)

Analysis of frozen Step 7 predictions on the official LIAR benchmark test split (`data/raw/test.tsv`, 1,283 samples).

### Test Performance
- **Accuracy:** 43.73%
- **Macro F1 Score:** 0.4345

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `likely_fake` | 0.3880 | 0.4357 | 0.4105 | 342 |
| `likely_reliable` | 0.4735 | 0.5043 | 0.4884 | 460 |
| `misleading` | 0.4401 | 0.3742 | 0.4045 | 481 |

### Top Error Confusion Patterns
| True Class | Predicted Class | Error Count | Share of Total Errors | Primary Root Cause |
| :--- | :--- | :---: | :---: | :--- |
| `misleading` | `likely_reliable` | 169 | 23.4% | Subtle context omission / half-truth boundary |
| `misleading` | `likely_fake` | 132 | 18.3% | Subtle context omission / half-truth boundary |
| `likely_reliable` | `misleading` | 125 | 17.3% | Subtle context omission / half-truth boundary |
| `likely_fake` | `misleading` | 104 | 14.4% | Subtle context omission / half-truth boundary |
| `likely_reliable` | `likely_fake` | 103 | 14.3% | Vocabulary overlap without external fact grounding |
| `likely_fake` | `likely_reliable` | 89 | 12.3% | Vocabulary overlap without external fact grounding |

---

## 5. Subsystem Verification Benchmarks

### A. Claim Extraction Benchmark (15 cases)
- **Checkability Accuracy:** 100.0%
- **Quote / Attribution Preservation:** 100.0%
- **Subjective Opinion Rejection Rate:** 100.0%

### B. Semantic Evidence Retrieval Benchmark (16 queries)
- **Recall@1:** 100.0%
- **Recall@3:** 100.0%
- **Mean Top-1 In-Domain Similarity:** 0.8101
- **No-Match Rejection Rate (Out-of-Domain):** 100.0%
- **False Retrieval Rate:** 0.0%

### C. Formal Fusion Safety Invariants (10 Invariants)
- **Total Invariants Tested:** 10
- **Passed Invariants:** 10 / 10 (100.0%)

| Invariant ID | Policy Name | Status |
| :--- | :--- | :---: |
| `INV-01` | No Checkable Claims Abstention | **PASSED** |
| `INV-02` | No Usable Evidence Abstention | **PASSED** |
| `INV-03` | Synthetic Fixture Quarantine in Production | **PASSED** |
| `INV-04` | Stylometric Red Flags Non-Conviction Gate | **PASSED** |
| `INV-05` | Classical ML Inability to Bypass Evidence Gate | **PASSED** |
| `INV-06` | LLM Reasoner Failure Graceful Fallback | **PASSED** |
| `INV-07` | LLM Unverifiable Assessment Honor Guard | **PASSED** |
| `INV-08` | Evidence Precedence Over Contrary ML (Reliable) | **PASSED** |
| `INV-09` | Evidence Precedence Over Contrary ML (Fake) | **PASSED** |
| `INV-10` | Strict Abstention Score Heuristic Equivalence | **PASSED** |

---

## 6. API & Frontend Verification Checklist

### FastAPI Backend Verification
- [x] `GET /healthz`: Returns HTTP 200 with online status and service metadata.
- [x] `GET /api/v1/health`: Returns detailed subsystem connectivity and model identifiers.
- [x] `POST /api/v1/analyze`: Processes statements up to 20,000 characters and emits full analysis schema.
- [x] `POST /api/v1/analyze` (Validation): Rejects empty strings, whitespace, and inputs > 20,000 chars with HTTP 422.
- [x] `GET /api/v1/analyses`: Paginated audit history browsing with `limit` and `offset` parameters.
- [x] `GET /api/v1/analyses/{id}`: Directly loads persisted audit record from SQLite without re-running ML/LLM pipeline.
- [x] `POST /api/v1/analyses/{id}/feedback`: Records user feedback (rating, notes, corrected verdict) with foreign key integrity (HTTP 201 Created).
- [x] `DELETE /api/v1/analyses/{id}`: Deletes record and cascaded feedback from SQLite database.

### Streamlit Frontend Smoke Check & UI Verification
- [x] **Smoke Test Startup:** `Streamlit server started on port 8501.`
- [x] **Character Counter:** Interactive UI counter tracks characters up to 20,000 limit with real-time feedback.
- [x] **Decision-Support Score Semantics:** Explicitly presented as **Decision-Support Strength** (`0.00 to 1.00`), never as a percentage of truth.
- [x] **Subsystem Disclaimers:** Visible disclaimers attached to Classical ML, Stylometric Flags, and Evidence Retrieval.
- [x] **Empty State Handling:** When zero evidence is found, UI clearly states: *"No usable provenance-bearing evidence was available in the current knowledge base."*
- [x] **Audit History View:** Reads directly from database without triggering background AI/ML pipelines.

---

## 7. Latency and Resource Utilization Profile

Measured across 50 iterations on local Windows workstation:

| Pipeline Subsystem | Median Latency (ms) | p95 Latency (ms) | Min Latency (ms) | Max Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `text_cleaning` | 0.001 | 0.001 | 0.001 | 0.006 |
| `stylometric_red_flags` | 0.058 | 0.084 | 0.055 | 0.113 |
| `classical_ml_inference` | 0.927 | 1.105 | 0.851 | 1.720 |
| `semantic_retrieval_demo_kb` | 12.086 | 30.780 | 7.996 | 400.036 |
| `hybrid_decision_fusion` | 0.014 | 0.016 | 0.013 | 0.057 |
| `sqlite_audit_insert` | 8.418 | 10.046 | 6.084 | 11.331 |
| `sqlite_audit_fetch` | 0.906 | 1.877 | 0.723 | 2.329 |
| `fastapi_testclient_e2e_overhead` | 16.517 | 84.301 | 11.163 | 109.052 |

- **Process Resident Set Size (RSS):** **739.62 MB** (within workstation boundaries).

---

## 8. Critical Research Insights & Honest Prototype Limitations

1. **LIAR Dataset Domain Shift:** Classical ML trained on political statements from 2007–2016 achieves 43.45% Macro F1. Language patterns alone fail on unseen modern domains and cannot verify physical real-world facts.
2. **Evidence Knowledge Base Coverage:** The production KB contains 0 verified records. The system's 100% abstention on out-of-KB statements is a safety feature, not a bug.
3. **Local Search Boundary:** The prototype intentionally lacks live web-scraping to prevent hallucinations and maintain reproducible academic evaluation.
4. **Score Interpretation:** Support scores reflect internal heuristic signal agreement, not calibrated truth probabilities.

---

## 9. Conclusion

The project has undergone rigorous multi-track evaluation as a functional academic prototype. All subsystems, fusion safety rules, API endpoints, and user interfaces behave deterministically and safely.

Step 19 complete. The project has been evaluated as a functional academic prototype. Ready for Step 20: final documentation, viva preparation, and submission packaging.