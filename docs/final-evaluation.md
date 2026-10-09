# Final System Evaluation & Prototype Analysis (Step 19)

**Project:** Fake News Detection Using LLMs  
**Project ID:** P_105  
**Student:** Bhagya Varshney  
**System Classification:** Functional Academic Prototype (Not a Global Fact-Checker)  
**Evaluation Date:** October 2026  

---

## 1. Prototype Classification & Evaluation Principles

This evaluation establishes the empirical, behavioral, and architectural profile of the system as an **academic prototype**.

### Core Evaluation Constraints & Disclosures:
1. **Clean Metric Separation:** Synthetic functional benchmark accuracy (Track A: 100.0%) is strictly separated from empirical LIAR benchmark test set accuracy (Track C: 43.73%). Synthetic accuracy measures internal fusion logic, not generalized real-world fake-news detection accuracy.
2. **Zero Model Retraining / Tuning:** The classical ML model (`models/classical_best_validation.joblib`) was frozen after validation tuning and evaluated only once on the official test set. No retraining, partial fitting, or hyperparameter adjustments were performed during or after evaluation.
3. **Empty Production Knowledge Base:** The production evidence store (`data/evidence/evidence_kb.jsonl`) contains 0 records (0 bytes). The system exhibits an intentional 100% abstention rate on out-of-KB statements.
4. **Uncalibrated Decision-Support Score:** `support_score` is an **uncalibrated heuristic decision-support score** (0.00 to 1.00) measuring evidentiary agreement and multi-signal corroboration. It is NOT an objective or calibrated probability of factual truth.
5. **Deterministic & Auditable:** All evaluation tracks execute completely offline with zero live Gemini API consumption and zero modification to production databases (`fake_news_analysis.db` remains at 0 rows).

---

## 2. Multi-Track Evaluation Summary

| Evaluation Track | Benchmark Dataset / Asset | Evaluated Target | Primary Finding / Metric |
| :--- | :--- | :--- | :--- |
| **Track A** | `data/evaluation/hybrid_controlled_eval.jsonl` (44 cases) | End-to-end multi-signal decision fusion across 15 behavioral categories | **100.0% Accuracy**, **1.0000 Macro F1**, **0 Unsafe Decisions** |
| **Track B** | `data/evaluation/production_abstention_eval.json` (20 statements) | Production mode behavior with empty local knowledge base | **100.0% Abstention Rate** (`unverifiable`, score 0.0), 100% short-circuit rate |
| **Track C** | `models/classical_test_predictions.csv` (1,283 test samples) | Frozen TF-IDF + Logistic Regression baseline on official LIAR test split | **43.73% Test Accuracy**, **0.4345 Test Macro F1** |
| **Claim Extraction** | `data/evaluation/claim_extraction_eval.json` (15 synthetic cases) | Atomic proposition isolation, checkability, quotes, and numbers | **100.0% Checkability Accuracy**, 100% quote preservation |
| **Evidence Retrieval**| `data/evaluation/retrieval_eval.json` (16 test queries) | Semantic retrieval on demo KB (`evidence_demo.jsonl`) | **100.0% Recall@1**, **100.0% Recall@3**, 100% OOD rejection |
| **Fusion Invariants** | 10 formal safety invariants | Strict evidence gating, synthetic quarantine, stylometric non-conviction | **10 / 10 Invariants Passed (100.0%)** |
| **System Latency** | 50 iterations on local workstation | Local subsystem inference and I/O latency | Classical ML median: **0.95 ms**, Retrieval median: **10.94 ms** |

---

## 3. Subsystem Benchmarks & Invariant Verification

### A. Fusion Safety Invariants (10 / 10 Passed)
- `INV-01 (No Checkable Claims)`: Subjective opinions and rhetorical questions strictly abstain (`unverifiable`, score 0.0).
- `INV-02 (No Usable Evidence)`: Factual propositions with 0 retrieved evidence strictly abstain (`unverifiable`, score 0.0).
- `INV-03 (Synthetic Quarantine)`: Synthetic demo fixtures are quarantined when `allow_synthetic_evidence=False`, forcing abstention.
- `INV-04 (Red Flags Non-Conviction)`: Heavy stylometric flags reflect writing style only and cannot independently convict without evidence.
- `INV-05 (ML Confidence Bounded)`: Even 98% ML confidence cannot override the evidence gate when evidence is missing.
- `INV-06 (LLM Failure Fallback)`: If evidence exists but LLM reasoning is unavailable, the system safely abstains.
- `INV-07 (LLM Unverifiable Guard)`: When the LLM reasoner assesses unverifiable, fusion strictly abstains with score 0.0.
- `INV-08 (Evidence Precedence - Reliable)`: Verified supporting evidence overrides contrary statistical ML predictions.
- `INV-09 (Evidence Precedence - Fake)`: Verified contradicting evidence overrides contrary statistical ML predictions.
- `INV-10 (Strict Score Equivalence)`: `abstained == True` strictly implies `support_score == 0.0` with zero exceptions.

### B. Latency and Resource Profile (50 Iterations)
- **Text Cleaning:** Median **0.001 ms**, p95 **0.001 ms**
- **Stylometric Red Flags:** Median **0.061 ms**, p95 **0.086 ms**
- **Classical ML Inference:** Median **0.948 ms**, p95 **1.110 ms**
- **Semantic Retrieval (Demo KB):** Median **10.937 ms**, p95 **72.543 ms**
- **Hybrid Decision Fusion:** Median **0.018 ms**, p95 **0.034 ms**
- **SQLite Audit Insert:** Median **9.143 ms**, p95 **12.642 ms**
- **SQLite Audit Fetch:** Median **0.861 ms**, p95 **1.788 ms**
- **FastAPI TestClient Overhead:** Median **14.463 ms**, p95 **58.389 ms**
- **Process Working Set (RSS):** **738.98 MB**

---

## 4. UI & Service Smoke Check Checklist

- [x] **FastAPI Health:** `GET /healthz` returns HTTP 200 with online status.
- [x] **Streamlit Startup:** `Streamlit server started on port 8501.`
- [x] **Input Boundary:** Enforces 20,000 character maximum across backend, frontend, and database schemas.
- [x] **Decision-Support Score Semantics:** UI labels metric as **Decision-Support Strength** (`0.00 to 1.00`), never as truth probability.
- [x] **Empty State Handling:** Clear messaging when zero evidence is retrieved.
- [x] **Audit History:** Loads stored analyses directly from SQLite without re-running ML or LLM pipelines.

---

## 5. Artifact Manifest

All generated evaluation outputs are stored in `reports/final_evaluation/`:
- `controlled_results.csv`: Row-by-row decisions for the 44 controlled benchmark scenarios.
- `controlled_metrics.json`: Precision, recall, F1, accuracy, and invariant checks.
- `controlled_confusion_matrix.csv`: 4x4 multi-signal confusion matrix.
- `production_behavior.csv`: Empirical responses for the 20 real-world production test statements.
- `production_behavior_summary.json`: Abstention rates, short-circuit telemetry, and ML distributions.
- `classical_test_metrics.json`: Frozen LIAR test set performance metrics.
- `classical_test_confusion_matrix.csv`: 3x3 classical confusion matrix.
- `claim_extraction_results.csv` & `claim_extraction_metrics.json`: Extraction functional benchmark.
- `retrieval_results.csv` & `retrieval_metrics.json`: Semantic retrieval evaluation metrics.
- `fusion_invariants.json`: Verification report for all 10 safety invariants.
- `performance_metrics.json`: Local latency and memory profiling statistics.
- `final_evaluation_report.md`: Complete synthesis report.

---

*Step 19 complete. The project has been evaluated as a functional academic prototype. Ready for Step 20: final documentation, viva preparation, and submission packaging.*
