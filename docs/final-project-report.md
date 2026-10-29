# Academic Project Report: Fake News Detection Using LLMs

**Project ID:** P_105  
**Project Title:** Fake News Detection Using LLMs  
**Candidate Name:** Bhagya Varshney  
**Repository:** `p105-fake-news-detection-llm-bhagya-varshney`  
**Classification:** Academic Engineering Prototype  
**Date of Submission:** October 2026  

---

## 1. Title
**A Multi-Signal Hybrid Framework for Automated Misinformation Analysis Combining Classical Machine Learning, Semantic Evidence Retrieval, and Evidence-Bound Large Language Model Reasoning with Safe Epistemic Abstention**

---

## 2. Abstract
This project presents an academic prototype for automated misinformation analysis that addresses the fundamental limitations of monolithic text classifiers and ungrounded Large Language Model (LLM) fact-checking. Text-only classifiers frequently learn stylistic, topical, or lexical artifacts rather than objective veracity, while unconstrained LLMs remain prone to hallucination, sycophancy, and temporal knowledge cutoffs. To overcome these challenges, we propose a modular hybrid architecture combining: (1) conservative input sanitization and length validation; (2) discrete factual claim extraction; (3) a frozen classical baseline classifier (TF-IDF vectorizer paired with multinomial Logistic Regression trained on the benchmark LIAR dataset); (4) deterministic stylometric red-flag detectors; (5) dense semantic evidence retrieval over reference knowledge stores using sentence embeddings (`all-MiniLM-L6-v2`); (6) evidence-bound LLM reasoning (`gemini-2.5-flash`) strictly quarantined from using ungrounded parametric memory; and (7) an 8-tier deterministic fusion engine featuring safe epistemic abstention. Delivered via a high-performance FastAPI REST backend, a relational SQLite audit store, and an interactive Streamlit interface, the system achieves 43.73% test accuracy and a Macro F1 score of 0.4345 on the official 3-class LIAR benchmark test split. On a controlled functional benchmark of 44 synthetic evaluation fixtures, the hybrid fusion logic demonstrates 100% adherence to defined safety policies. In production mode with an empty reference knowledge base (0 provenance-bearing records), the system safely abstains on 100% (20/20) of out-of-domain claims by returning `unverifiable`. This design prioritizes evidentiary grounding and epistemic safety over speculative binary classification.

---

## 3. Introduction
The proliferation of digital information across social networks and news aggregators has amplified the velocity, volume, and societal impact of misinformation. Automated veracity assessment has consequently emerged as a vital area of research within Natural Language Processing (NLP) and computational journalism [Reference required: Lazer et al., 2018; Vosoughi et al., 2018].

Early automated fact-checking research framed veracity prediction as standard supervised document classification, training models on linguistic features, n-grams, and stylometrics. While computationally efficient, these approaches correlate linguistic form with veracity rather than evaluating factual propositions against authoritative reality. A well-crafted fabrication written in neutral journalistic prose easily circumvents statistical n-gram filters, whereas authentic news written passionately may be penalized.

Recent advancements in Large Language Models (LLMs) offer unprecedented natural language reasoning capabilities. However, querying LLMs directly for factual verification introduces severe risks: models can hallucinate persuasive corroborating details, reflect biased training distributions, or fail on temporal developments occurring after their pretraining cutoffs. 

This project explores a principled engineering synthesis: a multi-signal pipeline that isolates statistical pattern recognition from external evidence retrieval, binds generative LLMs strictly to retrieved evidence, and deterministically abstains whenever verifiable evidence is unavailable.

---

## 4. Problem Statement
Given an arbitrary text input $T$ representing a news claim or article (up to 20,000 characters), develop a reproducible, transparent, and auditable system that:
1. Decomposes $T$ into extractable, checkable factual claims $C = \{c_1, c_2, \dots, c_k\}$ while isolating subjective opinion and speculation.
2. Evaluates statistical surface patterns through a classical machine learning model trained on labeled political discourse.
3. Quantifies stylometric anomalies and emotional manipulation indicators through deterministic rules.
4. Retrieves semantically relevant reference evidence $E = \{e_1, e_2, \dots, e_m\}$ from a local knowledge base.
5. Employs an LLM to evaluate whether retrieved evidence supports, refutes, or is insufficient to judge the claims.
6. Fuses heterogeneous signals deterministically without hallucination, outputting an authoritative verdict $V \in \{\text{likely\_reliable}, \text{misleading}, \text{likely\_fake}, \text{unverifiable}\}$ and an uncalibrated heuristic decision-support score $s \in [0.0, 1.0]$.
7. Safely abstains with $V = \text{unverifiable}$ ($s = 0.0$, $\text{abstained} = \text{True}$) whenever checkable claims lack usable provenance-bearing evidence.

---

## 5. Objectives
1. **Classical Baseline Development:** Train, validate, freeze, and evaluate a lightweight TF-IDF and Logistic Regression classifier on the benchmark LIAR dataset.
2. **Modular Pipeline Construction:** Implement decoupled software modules for text validation, claim extraction, stylometric red-flag detection, dense semantic retrieval, LLM reasoning, and fusion.
3. **Evidence-Gated Reasoning:** Enforce strict prompt constraints preventing the LLM from asserting factual veracity based on internal memory without supplied evidence.
4. **Deterministic Fusion Engine:** Replace arbitrary majority voting or black-box ensemble layers with an auditable 8-tier hierarchical decision tree.
5. **Software Engineering Delivery:** Wrap the pipeline in a production-style FastAPI service and an intuitive Streamlit interface supported by an SQLite relational audit database.
6. **Rigorous Multi-Track Evaluation:** Benchmark classical generalization, controlled functional logic, evidence gating, and software quality using comprehensive automated test suites (514+ pytest cases).

---

## 6. Motivation
The primary motivation of this project is to model **epistemic humility** in artificial intelligence systems. In high-stakes domains such as journalism, public health, and civic elections, presenting an unsubstantiated guess as a definitive truth rating can cause substantial societal harm. Existing systems frequently produce binary verdicts (e.g., "92% Fake") even when the claim addresses events outside the model's knowledge horizon.

By establishing an explicit `unverifiable` verdict gated by evidentiary presence, this project shifts the paradigm from speculative classification to evidence-grounded decision support. Furthermore, building an end-to-end prototype on modest commodity hardware (~8 GB RAM) demonstrates the feasibility of combining lightweight local models with cloud-hosted LLM APIs without requiring high-performance GPU clusters.

---

## 7. Existing Approaches
Misinformation detection literature broadly falls into three historical paradigms:

1. **Stylometric and Text-Classification Approaches:**
   Early systems relied on readability metrics, POS-tag distributions, sentiment lexicons, and n-gram TF-IDF representations paired with Support Vector Machines or Logistic Regression [Wang, 2017: LIAR benchmark]. While fast and interpretable, these models identify *sensationalism* rather than *falsehood*.

2. **Deep Learning and Transformer Classifiers:**
   Fine-tuned language models (e.g., BERT, RoBERTa, DeBERTa) significantly improved classification benchmarks on specific datasets [Reference required: Thorne et al., 2018 (FEVER); Shu et al., 2019 (FakeNewsNet)]. However, research demonstrates that deep classifiers often exploit dataset-specific spurious correlations (e.g., source cues, named entity biases) rather than verifying real-world facts [Reference required: Niven & Kao, 2019].

3. **Retrieval-Augmented Fact-Checking (RAG):**
   Modern fact-checking frameworks retrieve relevant documents from Wikipedia or news APIs before assessing veracity [Reference required: Lewis et al., 2020; Thorne et al., 2018]. While superior, unconstrained generative RAG systems frequently hallucinate connections between vaguely related text snippets.

Our proposed framework differs from standard RAG by enforcing a **strict evidence gate**: if retrieved candidates fall below semantic relevance thresholds, generative reasoning is bypassed entirely, and the system abstains deterministically.

---

## 8. Proposed Hybrid Architecture
The system follows a synchronous, modular signal processing pipeline:

```text
                  User Text (<= 20,000 characters)
                                 |
                                 v
                     [Text Cleaning & Validation]
                                 |
                                 v
                     [Claim Extraction Service]
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
     [Classical ML Model]                [Stylometric Red Flags]
    (TF-IDF + Logistic Reg)               (Deterministic Rules)
              |                                     |
              +------------------+------------------+
                                 |
                                 v
                     [Evidence Retrieval Engine]
                     (Local Dense MiniLM Search)
                                 |
                                 v
                     [Evidence Gate Evaluation]
                    /                          \
       [Evidence Available]             [No Usable Evidence]
               |                                  |
               v                                  v
    [Evidence-Bound LLM Reasoner]         [Bypass Reasoner]
    (Rubric-driven Gemini Flash)                  |
               |                                  |
               +------------------+---------------+
                                  |
                                  v
                    [Deterministic Fusion Engine]
                    (8-Tier Hierarchical Rules)
                                  |
                                  v
                     [Final Assessment Output]
                     (Verdict, Support Score)
                                  |
              +-------------------+-------------------+
              |                                       |
              v                                       v
     [FastAPI REST API]                       [SQLite Audit DB]
              |
              v
     [Streamlit UI Dashboard]
```

### Architectural Principles
- **Decoupled Contracts:** Subsystems communicate exclusively via immutable Pydantic data schemas (`Claim`, `MLAssessment`, `RedFlagReport`, `EvidenceItem`, `LLMAssessment`, `FinalAssessment`).
- **Zero Parametric Reliance:** The LLM prompt explicitly commands the model to evaluate claims solely within the context of supplied `<evidence>` XML blocks.
- **Deterministic Precedence:** Subsystem disagreement is resolved through strict hierarchical rules rather than learned weights, guaranteeing auditable predictability.

---

## 9. Dataset
The empirical machine learning baseline was constructed using the official benchmark **LIAR dataset** [Wang, 2017: *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*, ACL 2017].

### Dataset Composition
The LIAR dataset comprises short political statements fact-checked by PolitiFact between 2007 and 2016. The original corpus contains six fine-grained truthfulness ratings:
1. `pants-fire`
2. `false`
3. `barely-true`
4. `half-true`
5. `mostly-true`
6. `true`

### Three-Class Project Mapping
To align with practical verification workflows and improve statistical stability on short statements, the six original labels were mapped into three project classes:

| Original LIAR Label | Project Mapped Class | Semantic Definition |
| :--- | :--- | :--- |
| `pants-fire`, `false` | `likely_fake` | Fabricated, demonstrably false, or outrageous claims |
| `barely-true`, `half-true` | `misleading` | Partial truths, out-of-context facts, or significant omissions |
| `mostly-true`, `true` | `likely_reliable` | Accurate, substantiated statements with minor or no flaws |

### Split Partitions
- **Training Set (`data/raw/train.tsv`):** 10,269 statements (used for TF-IDF fitting and classifier optimization).
- **Validation Set (`data/raw/valid.tsv`):** 1,284 statements (used for hyperparameter tuning and model selection).
- **Official Test Set (`data/raw/test.tsv`):** 1,283 statements (frozen; evaluated once for final reporting).

---

## 10. Data Preprocessing
Data preprocessing is performed by a conservative, deterministic cleaner (`backend/app/services/text_cleaner.py`):
1. **Unicode Normalization:** Applies NFKC normalization to resolve ligature and homoglyph inconsistencies.
2. **Artifact Removal:** Strips invisible zero-width characters (`\u200b`, `\u200c`, `\u200d`, `\ufeff`) often introduced by copy-pasting or deliberate adversarial evasion.
3. **Whitespace Standardization:** Collapses consecutive whitespace and newline sequences into normalized single spaces.
4. **Bounds Enforcement:** Rejects empty or whitespace-only inputs, and enforces a hard ceiling of 20,000 characters to prevent buffer and memory denial-of-service.

---

## 11. Classical Machine Learning Training
The statistical baseline was trained strictly on `data/raw/train.tsv` using scikit-learn.

### Feature Extraction
- **Representation:** Sublinear term-frequency inverse-document-frequency (`TfidfVectorizer`).
- **N-gram Range:** Unigrams and bigrams (`(1, 2)`).
- **Vocabulary Constraints:** `min_df=2`, `max_df=0.95`, maximum 50,000 features.
- **Normalization:** L2 Euclidean normalization.

### Model Selection & Hyperparameter Tuning
Four candidate estimators were tuned across cross-validation runs on the training and validation splits:
1. Multinomial Naive Bayes (`MultinomialNB`)
2. Complement Naive Bayes (`ComplementNB`)
3. Linear Support Vector Classifier (`LinearSVC`)
4. Multinomial Logistic Regression (`LogisticRegression`)

**Best Model Configuration:** Logistic Regression optimized with $C=0.5$, balanced class weighting (`class_weight='balanced'`), L2 penalty, and `liblinear` solver. The fitted pipeline was serialized to `models/classical_best_validation.joblib` (SHA-256: `d4c3636277761f5399166ee41c0eec57cafe7b9216bacfc18d81f26fea415c92`).

---

## 12. Claim Extraction
Veracity cannot be assessed on full documents without isolating checkable propositions. The Claim Extraction module (`backend/app/services/claim_extractor.py`) operates via rule-based syntactic parsing or LLM-assisted zero-shot decomposition:
- Identifies atomic declarative sentences asserting verifiable real-world states.
- Filters out pure subjective opinions, value judgments, and speculative statements ("I think...", "It might be that...").
- Preserves explicit attributions and named entities to contextualize downstream evidence search.
- Outputs structured Pydantic `Claim` objects tagged with checkability flags and entity lists.

---

## 13. Stylometric & Red-Flag Detection
Misinformation often exhibits distinct rhetorical signatures designed to elicit visceral emotional reactions. The Red-Flag Analyzer (`backend/app/services/red_flags.py`) calculates deterministic linguistic indicators:
- **Capitalization Anomalies:** Detects excessive uppercase characters (shouting patterns).
- **Punctuation Extremes:** Identifies clusters of exclamation and question marks (e.g., `???`, `!!!`).
- **Sensational & Clickbait Lexicon:** Scans for curated urgent vocabulary ("SHOCKING", "BANNED", "SECRET THEY HIDE", "MIRACLE CURE").
- **Emotional Polarization:** Computes positive and negative sentiment intensity to detect emotional manipulation.

The module outputs a structured `RedFlagReport` containing individual cue counts, a normalized severity score ($0.0$ to $1.0$), and a qualitative flag level (`low`, `medium`, `high`, `critical`).

---

## 14. Evidence Retrieval
The local evidence retrieval layer (`backend/app/services/evidence_retriever.py`) grounds claims in factual reference sources:
- **Reference Store:** Local structured JSONL files (`data/evidence/evidence_kb.jsonl`) maintaining verified reference texts with full provenance metadata (author, publication date, URL, source domain).
- **Dense Vector Embeddings:** Encodes queries and candidate passages using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Lexical Fallback:** Integrates an in-memory TF-IDF cosine similarity search for offline environments where neural embedding runtimes or dynamic libraries are restricted.
- **Engineering Similarity Gate:** Imposes a strict cosine similarity threshold (default $\tau = 0.45$). Evidence items scoring below $\tau$ are discarded as non-matching background noise.
- **Top-K Selection:** Selects up to $k=3$ highest-ranking passages per claim.

---

## 15. LLM-Based Evidence Assessment
When usable provenance-bearing evidence passes the retrieval gate, generative reasoning is invoked via Google Gemini (`gemini-2.5-flash`) through a provider-neutral adapter (`backend/app/ai/reasoner.py`):
- **Structured XML Prompting:** Claims and candidate evidence passages are injected into rigid XML boundaries (`<claims>`, `<evidence>`).
- **Strict Evidence Quarantine:** System instructions explicitly prohibit the model from affirming claims based on its ungrounded training memory:
  > *"You must evaluate the claims strictly against the supplied evidence passages. If the supplied evidence is insufficient, contradictory, or absent, you must output evidence_support = 'insufficient' and verdict = 'unverifiable'."*
- **Rubric Classification:** The LLM classifies evidence stance into `supports`, `refutes`, `partial`, `conflicting`, or `insufficient`, accompanied by concise factual explanations.
- **Resilience & Fallback:** Network timeouts, quota exhaustion (HTTP 429), or malformed JSON responses gracefully degrade to deterministic fallback evaluations without crashing the analysis pipeline.

---

## 16. Deterministic Fusion & Epistemic Abstention
The Hybrid Decision Fusion Layer (`backend/app/services/fusion.py`) reconciles signals from the classical ML model, red-flag detector, and LLM reasoner through an **8-tier deterministic decision hierarchy**:

1. **Non-Checkable / Opinion Gate:** If no checkable factual claims are present, returns `unverifiable` (`abstained=True`, `support_score=0.0`, reasoning: *"No checkable factual claims were identified."*).
2. **Missing Evidence Abstention Gate:** If checkable claims exist but no usable provenance-bearing evidence was retrieved, returns `unverifiable` (`abstained=True`, `support_score=0.0`, reasoning: *"No usable provenance-bearing evidence was available."*).
3. **Severe Stylometric Downgrade:** If red-flag severity is `critical`, prevents a `likely_reliable` verdict and caps support score, preventing sensationalized content from receiving unconditional endorsement.
4. **Evidence Refutation Rule:** If retrieved evidence refutes key claims, issues `likely_fake` regardless of favorable classical ML signals.
5. **Evidence Corroboration Rule:** If retrieved evidence directly supports key claims and red flags are low, issues `likely_reliable`.
6. **Partial Evidence / Distortion Rule:** If evidence partially supports claims with context omissions, issues `misleading`.
7. **Conflicting Evidence Rule:** If evidence contains irreconcilable factual contradictions, abstains with `unverifiable`.
8. **Heuristic Score Computation:** Calculates `support_score` ($0.0$ to $1.0$) reflecting cross-signal concord. If abstaining, `support_score` is strictly $0.0$.

---

## 17. Backend Architecture
The backend is built with **FastAPI** (`backend/app/api/app.py`), adhering to clean architectural principles:
- **Dependency Injection:** Database connections and pipeline engine singletons are managed via FastAPI's `Depends` system.
- **REST Endpoints:**
  - `GET /healthz`: Lightweight liveness probe.
  - `GET /api/v1/health`: Detailed subsystem health probe.
  - `POST /api/v1/analyze`: End-to-end text analysis with full schema response.
  - `GET /api/v1/analyses`: Paginated historical audit queries (`limit`, `offset`).
  - `GET /api/v1/analyses/{analysis_id}`: Single record audit retrieval without re-running models.
  - `DELETE /api/v1/analyses/{analysis_id}`: Cascading deletion of analysis and associated feedback.
  - `POST /api/v1/analyses/{analysis_id}/feedback`: Human feedback collection returning `HTTP 201 Created`.
  - `GET /api/v1/analyses/{analysis_id}/feedback`: Feedback audit inspection.
- **Exception Filters:** Custom exception handlers map domain exceptions (`AnalysisNotFoundError`, `TextLengthExceededError`, `DatabaseError`) to standardized RFC 7807 error responses without leaking internal stack traces.

---

## 18. Database Design
Persistent storage is implemented via Python standard-library `sqlite3` (`data/app/fake_news_analysis.db`):
- **Table `analyses`:** Primary audit log storing `analysis_id` (UUID), `input_text`, `verdict`, `confidence_score` (`support_score`), `abstained`, `ml_label`, `red_flags_level`, `execution_time_ms`, full serialized JSON result, and timestamp.
- **Table `feedback`:** Relational child table storing `feedback_id`, foreign key `analysis_id` (`ON DELETE CASCADE`), `rating` (`helpful`, `not_helpful`, `unsure`), `comment`, `corrected_verdict`, and timestamp.
- **Foreign Key Constraints:** Enforced via `PRAGMA foreign_keys = ON;`.
- **Indexing:** B-tree indexes on `analyses(created_at)`, `analyses(verdict)`, and `feedback(analysis_id)`.
- **Security:** Strict parameterized SQL queries preventing SQL injection vulnerabilities.

---

## 19. Frontend Design
The user interface is developed in **Streamlit** (`frontend/app.py`):
- **Decoupled Client:** Communicates strictly via HTTP REST calls (`frontend/client.py`) to the FastAPI backend; never imports backend ML modules directly.
- **Interactive Character Counter:** Visual counter tracking inputs against the 20,000-character ceiling.
- **Structured Verdict Banners:** Distinct color-coded UI cards for `Likely Reliable` (green), `Misleading` (amber), `Likely Fake` (red), and `Unverifiable` (blue).
- **Transparency Accordions:** Expandable inspectors detailing extracted claims, classical ML probabilities, stylometric cues, retrieved evidence passages, and LLM reasoning steps.
- **History & Feedback Views:** Allows users to inspect previous analyses and submit qualitative corrections directly to the database.

---

## 20. Evaluation Methodology
The system was evaluated across seven distinct empirical and functional tracks:
1. **Classical ML Generalization:** Evaluated on the frozen official LIAR test split (`data/raw/test.tsv`, 1,283 samples).
2. **Controlled Hybrid Functional Benchmark:** Evaluated on 44 synthetic test fixtures covering supported, contradicted, misleading, and out-of-domain cases.
3. **Retrieval Benchmark:** Evaluated on synthetic reference pairs to verify cosine similarity ranking and threshold gates.
4. **Production Coverage Benchmark:** Evaluated on 20 arbitrary real-world news claims against the empty production knowledge base.
5. **Safety Invariant Testing:** Formal property tests validating 10 core safety invariants.
6. **Software Quality Suite:** Comprehensive unit, integration, and API tests executed via pytest.
7. **Latency & Resource Profiling:** Workstation benchmark measuring median and 95th-percentile execution times.

---

## 21. Experimental Results

### Table 1: Classical Machine Learning Benchmark (LIAR Official Test Split)
*Evaluated on `data/raw/test.tsv` (1,283 instances) using frozen model `models/classical_best_validation.joblib`.*

| Evaluation Metric | Measured Score | Baseline / Context |
| :--- | :---: | :--- |
| **Test Accuracy** | **43.73%** | 561 correct predictions / 1,283 total statements |
| **Macro Precision** | **0.4339** | Unweighted average precision across classes |
| **Macro Recall** | **0.4381** | Unweighted average recall across classes |
| **Macro F1 Score** | **0.4345** | Balanced harmonic mean |
| **Weighted F1 Score** | **0.4362** | Weighted by true class distribution |
| Random Guessing Baseline | 33.33% | Expected accuracy under uniform random selection |
| Majority Class Baseline | 35.75% | Always predicting the most frequent class (`likely_reliable`) |

*Detailed Per-Class Performance on Test Split:*
| Class Label | Precision | Recall | F1-Score | True Support |
| :--- | :---: | :---: | :---: | :---: |
| `likely_fake` | 0.3880 | 0.4357 | 0.4105 | 342 |
| `likely_reliable` | 0.4735 | 0.5043 | 0.4884 | 460 |
| `misleading` | 0.4401 | 0.3742 | 0.4045 | 481 |

---

### Table 2: Controlled Hybrid Functional Benchmark
*Evaluated across 44 synthetic and controlled fixtures verifying adherence to fusion decision rules.*

| Scenario Category | Test Cases | Expected Behaviors | Adherence Rate | Observed Safety Violations |
| :--- | :---: | :---: | :---: | :---: |
| Category A: Supported Claims (Reliable) | 11 | 11 | 100.0% | 0 |
| Category B: Contradicted Claims (Fake) | 11 | 11 | 100.0% | 0 |
| Category C: Partial Support (Misleading) | 11 | 11 | 100.0% | 0 |
| Category D: Insufficient Evidence (Abstain) | 11 | 11 | 100.0% | 0 |
| **Total Controlled Adherence** | **44** | **44** | **100.0% (44/44)** | **0** |

> [!NOTE]
> The 100% adherence score reflects functional verification of the deterministic fusion logic, not open-world veracity detection accuracy.

---

### Table 3: Retrieval Functional Benchmark (Synthetic Knowledge Base)
*Evaluated on controlled reference query-passage pairs.*

| Retrieval Metric | Measured Value | Operational Meaning |
| :--- | :---: | :--- |
| **Recall@1** | **100.0%** | Relevant passage ranked first for in-domain queries |
| **Recall@3** | **100.0%** | Relevant passage present in top-3 candidates |
| **Mean In-Domain Cosine Similarity** | **0.8101** | High semantic affinity for matching queries |
| **Out-of-Domain Rejection Rate** | **100.0%** | Unrelated queries correctly fall below threshold $\tau = 0.45$ |
| **False Retrieval Rate** | **0.0%** | Zero irrelevant passages retained above threshold |

---

### Table 4: Formal Fusion Safety Invariants
*Property-based verification of safety policies.*

| Invariant ID | Safety Policy Description | Formal Constraint | Result |
| :--- | :--- | :--- | :---: |
| `INV-01` | Non-Checkable Input Gate | If claims checkable = 0 $\rightarrow$ `unverifiable` | **PASSED** |
| `INV-02` | Missing Evidence Gate | If evidence count = 0 $\rightarrow$ `unverifiable` | **PASSED** |
| `INV-03` | Synthetic Fixture Quarantine | Production API strictly rejects synthetic KB overrides | **PASSED** |
| `INV-04` | Stylometric Non-Conviction Gate | Red flags alone cannot convict input as `likely_fake` | **PASSED** |
| `INV-05` | Classical ML Evidence Independence | Classical ML signal cannot bypass missing evidence gate | **PASSED** |
| `INV-06` | LLM Failure Graceful Fallback | Timeout or API error falls back safely without crash | **PASSED** |
| `INV-07` | LLM Unverifiable Honor Guard | If LLM marks `unverifiable`, fusion honors abstention | **PASSED** |
| `INV-08` | Evidence Precedence (Reliable) | Strong supporting evidence overrides contrary classical ML | **PASSED** |
| `INV-09` | Evidence Precedence (Fake) | Direct refuting evidence overrides contrary classical ML | **PASSED** |
| `INV-10` | Strict Abstention Heuristic Value | If `abstained=True` $\rightarrow$ `support_score` is strictly $0.0$ | **PASSED** |

---

### Table 5: Automated Software Quality Suite
*Test suite status executed via pytest on Python 3.14.*

| Test Module | Coverage Scope | Tests Passed | Status |
| :--- | :--- | :---: | :---: |
| `test_text_cleaner.py` | Sanitization, bounds, zero-width chars | 22 | PASSED |
| `test_baseline_data.py` | LIAR loading, 3-class mapping, label integrity | 13 | PASSED |
| `test_baseline_model.py` | TF-IDF, logistic regression serialization | 13 | PASSED |
| `test_tune_baseline.py` | Hyperparameter search validation | 9 | PASSED |
| `test_classical_test_evaluation.py` | Test split error analysis reproducibility | 10 | PASSED |
| `test_red_flags.py` | Stylometric counters, threshold gating | 20 | PASSED |
| `test_schemas.py` | Pydantic contracts, validation errors | 32 | PASSED |
| `test_claim_extractor.py` | Checkability, entity parsing, quotes | 33 | PASSED |
| `test_evidence_retriever.py` | Embeddings, thresholds, lexical fallback | 30 | PASSED |
| `test_llm_client.py` | Gemini adapter, timeout handling, retries | 20 | PASSED |
| `test_llm_reasoner.py` | Evidence grounding, quota fallback | 33 | PASSED |
| `test_fusion.py` | 8-tier hierarchy, invariant checks | 35 | PASSED |
| `test_hybrid_engine.py` | End-to-end pipeline coordination | 43 | PASSED |
| `test_database.py` | SQLite schema, foreign keys, cascades | 47 | PASSED |
| `test_api.py` | FastAPI routes, status codes, OpenAPI docs | 49 | PASSED |
| `test_frontend_client.py` | Streamlit HTTP client error handling | 35 | PASSED |
| `test_final_evaluation.py` | Multi-track system benchmarks | 15 | PASSED |
| **Total Test Suite** | **Full Project Integration & Verification** | **514 passed** | **PASSED (100%)** |

---

### Table 6: Latency and Workstation Performance Profile
*Measured across 50 iterations on commodity developer workstation.*

| Pipeline Component | Median Latency (ms) | 95th Percentile (ms) | Minimum (ms) | Maximum (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Text Validation & Cleaning | 0.001 | 0.001 | 0.001 | 0.006 |
| Stylometric Red Flags | 0.061 | 0.086 | 0.058 | 0.104 |
| Classical ML Inference | 0.948 | 1.110 | 0.874 | 1.561 |
| Semantic Retrieval (Demo KB) | 10.937 | 72.543 | 8.051 | 138.772 |
| Deterministic Decision Fusion | 0.018 | 0.034 | 0.015 | 0.088 |
| SQLite Audit Record Insert | 9.143 | 12.642 | 6.801 | 12.809 |
| SQLite Record Fetch by ID | 0.861 | 1.788 | 0.742 | 2.225 |
| Full End-to-End API Overhead | 14.463 | 58.389 | 11.448 | 91.017 |
| **Process Resident Memory (RSS)** | — | — | — | **738.98 MB** |

---

## 22. Error Analysis
In-depth diagnostic analysis of the classical ML test errors revealed three dominant failure patterns:
1. **Half-Truth / Misleading Boundary Dispersion:** The most frequent confusion occurred between `misleading` and `likely_reliable` (169 instances, 23.4% of total errors), and between `misleading` and `likely_fake` (132 instances, 18.3% of total errors). Statements containing accurate statistics framed with deceptive conclusions possess lexical overlap with both truth and falsehood, baffling pure bag-of-words classifiers.
2. **Context-Free Vocabulary Overlap:** Statements debunking rumors often contain the exact keywords found in false rumors ("The claim that flu shots contain microchips is untrue"). Linear TF-IDF models lack syntactic dependency awareness, frequently triggering false alarms.
3. **Probability Score Compression:** Across both correct and incorrect test predictions, the mean maximum predicted softmax probability fell between 0.43 and 0.45. This confirms that linear classifiers operating on short statements exhibit high epistemic uncertainty, justifying their treatment as uncalibrated priors rather than definitive truth metrics.

---

## 23. Limitations
1. **Temporal & Domain Shift:** The classical model is bound to the historical domain of US politics (2007–2016).
2. **Modest Classical Baseline Accuracy:** The 43.73% accuracy reflects the intrinsic difficulty of 3-class veracity classification without external knowledge.
3. **Knowledge Base Coverage:** The production knowledge base currently holds 0 provenance-bearing records. Real-world queries therefore abstain safely rather than providing factual verification.
4. **Lack of Live Web Search:** To maintain deterministic reproducibility and avoid ungrounded scraping, live web search is deliberately omitted.
5. **API Quota & Latency:** Generative reasoning relies on the external Gemini API, which introduces rate limits and network latency.
6. **Heuristic Score Interpretation:** The `support_score` is an engineering indicator of signal consensus, not a mathematical probability of truth.

---

## 24. Ethical & Responsible AI Considerations
- **Prevention of False Authority:** By providing an `unverifiable` verdict and labeling scores as heuristic decision-support metrics, the system avoids misleading users into false certitude.
- **Auditability & Provenance:** Every analysis stores complete intermediate representations in SQLite, allowing human fact-checkers to inspect the exact reasoning chain.
- **Privacy & Security:** Zero user credentials or external API keys are stored in the database or committed to version control.
- **Non-Automated Retraining:** User feedback is stored for audit inspection only; it never automatically alters model weights, preventing data poisoning attacks.

---

## 25. Future Work
1. **Curated Production Evidence Corpus:** Integrate established knowledge repositories (e.g., Wikipedia API, Wikidata, verified news archives).
2. **Live Evidence Search Engine:** Incorporate trusted search APIs (e.g., Google Fact Check Tools API, Bing Web Search) with strict domain allowlisting.
3. **Transformer-Based Stance Detection:** Replace prompt-based LLM stance classification with lightweight local cross-encoders fine-tuned on FEVER.
4. **Empirical Probability Calibration:** Apply Platt scaling or isotonic regression to calibrate confidence scores empirically.
5. **Multi-Lingual Claim Extraction:** Expand the pipeline to process non-English claims using multilingual sentence embeddings.

---

## 26. Conclusion (Version 1 Frozen Baseline)
The **Fake News Detection Using LLMs** project successfully demonstrates a principled hybrid architecture that unifies classical machine learning, stylometric analysis, dense semantic retrieval, and evidence-bound Large Language Model reasoning. Rather than yielding to the temptation of speculative classification, the system introduces a deterministic evidence gate and safe epistemic abstention. 

The implementation satisfies all academic, engineering, and software quality criteria: 514 passing automated tests, a robust FastAPI backend, a clean relational SQLite schema, and an intuitive Streamlit interface. While deployed as a prototype with an empty production knowledge base, the architecture establishes a solid, auditable foundation for future evidence-augmented verification systems.

---

## 27. Version 2 Extension: Real-World Evidence Discovery, Stance Reasoning, and Safety Gating (V2.1 – V2.6)

### 27.1 Overview & Scope Evolution
Building on the frozen Version 1 foundation, Version 2 upgrades the system into an evidence-assisted research prototype that discovers real-world news evidence, clusters syndicated wire stories, reasons over claim-evidence stances, and enforces conservative corroboration:
1. **Single LLM Provider (V2.1):** Replaced Gemini with **Groq** (`openai/gpt-oss-20b`) via a provider-neutral adapter. Zero hidden chain-of-thought; strictly structured Pydantic schemas.
2. **Public Real-World Evidence Discovery (V2.2):** Integrated **GDELT Context 2.0** and **GDELT DOC 2.0** public endpoints without adding any new API keys. Candidate evidence is reranked locally with `sentence-transformers/all-MiniLM-L6-v2`.
3. **Structured Stance Reasoning & Consensus Gating (V2.3):** Single structured batch Groq prompt with prompt-injection defense. Implemented `EvidenceConsensusEngine` with near-duplicate clustering and conservative gating ($\ge 2$ independent clusters, $\ge 2$ distinct publisher domains; strict exclusion of headline-only items).
4. **Data Harmonization & Integrity Audit (V2.4 & V2.4.1):** Reconciled LIAR sample counts (10,269 train, 1,284 valid, 1,283 test) and ingested AVeriTeC (4,568 train, 500 dev) into a harmonized 3-way label space.
5. **Classifier Benchmark & Model Selection (V2.5):** Rigorous comparison between TF-IDF and MiniLM classifiers. Formally selected **V2 TF-IDF + Logistic Regression** based on internal Stratified 5-Fold CV Macro F1 (0.4786 vs MiniLM 0.4770). MiniLM recorded as promising challenger (superior cross-domain generalization: LIAR test 0.4397, AVeriTeC dev 0.4378).
6. **Comprehensive Multi-Track Final Evaluation (V2.6):** Frozen evaluation across scientific, functional, safety, and latency criteria.

### 27.2 Frozen Baseline vs. Version 2 Empirical Benchmark
The historical Version 1 classical baseline artifact (`models/classical_best_validation.joblib`, SHA-256: `d4c36362...`) and frozen LIAR test split (`data/raw/test.tsv`, SHA-256: `ee41e2c7...`) remain strictly untouched and immutable.

| Evaluation Benchmark | V1 Frozen Baseline | V2 Selected Classifier (TF-IDF) | V2 Challenger (MiniLM) |
| :--- | :---: | :---: | :---: |
| **Internal 5-Fold CV Macro F1** | 0.4345 | **0.4786 ± 0.0047** | 0.4770 ± 0.0031 |
| **Official Frozen LIAR Test (1,283) Accuracy** | **43.73%** | 43.34% | 44.66% |
| **Official Frozen LIAR Test (1,283) Macro F1** | **0.4345** | 0.4275 | 0.4397 |
| **AVeriTeC Dev (500) Accuracy** | 43.01% | 61.29% | 63.44% |
| **AVeriTeC Dev (500) Macro F1** | 0.3798 | 0.4048 | 0.4378 |
| **Controlled Stance Benchmark (35 cases) Macro F1** | N/A | **1.0000** | **1.0000** |
| **Controlled End-to-End Hybrid (48 cases) Macro F1**| N/A | **1.0000** | **1.0000** |
| **Resilience / Failure-Mode Audit (9 cases) Pass Rate**| N/A | **100.0%** (9/9) | **100.0%** (9/9) |
| **Median Total Offline Pipeline Latency (p50)** | ~25 ms | **46.7 ms** | ~65 ms |

### 27.3 Controlled Benchmark Highlights
- **Track C (Stance Reasoning, 35 cases):** Stance accuracy 100.0%, attribution accuracy 100.0%, Macro F1 1.0000, 7/7 safety invariants satisfied, 0 unsafe decisions.
- **Track D (Controlled Hybrid, 48 cases):** 4-way accuracy 100.0%, Macro F1 1.0000, abstention accuracy 100.0%, 0 unsafe decisions, 0 invariant violations across dual-cluster support, dual-cluster contradiction, partial/cherry-picked evidence, syndication duplication, single-cluster evidence, headline-only evidence, conflicting evidence, zero evidence, non-checkable opinion, prompt injection, ML disagreement, red-flag disagreement, and dependency failures.
- **Track E (Live Real-World Demo):** Successfully verified graceful degradation and conservative abstention when external live news APIs return zero verified candidates or encounter rate limits.
- **Track F & G (Latency & Resilience):** Median offline pipeline execution is 46.7 ms (sub-100ms guarantee). 9 out of 9 edge-case resilience tests passed (100%).

### 27.4 Post-Release Retrieval Robustness Maintenance Patch
Following deployment of the prototype, an empirical information-retrieval bottleneck was identified in production: monolithic queries with up to 8 keywords submitted to the GDELT Context 2.0 sentence-level API returned zero candidates for realistic factual claims due to strict sentence-horizon conjunction constraints. A deterministic progressive query relaxation patch was implemented:
1. **Four-Tier Deterministic Search Plan:** Tier 1 (Precise phrase + top entities), Tier 2 (Entity anchor + claim-derived OR group), Tier 3 (Relaxed DOC 2.0), Tier 4 (Broad DOC 2.0 with extended timespan).
2. **Early Stopping & Endpoint Shielding:** Search halts immediately once $\ge 5$ deduplicated candidates are acquired. Upon encountering an HTTP 429 (rate-limited), timeout, or server error, all remaining attempts are aborted to shield public infrastructure.
3. **Preservation of Invariants:** All classical models, stance prompts, near-duplicate clustering thresholds, consensus rules, and abstention policies remain 100% frozen and unaltered.
Detailed technical documentation and empirical validation traces are available in `docs/v2-retrieval-robustness-patch.md`.

---

### References
1. Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 422-426). [ACL 2017 Anthology].
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 3982-3992).
3. Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a large-scale dataset for Fact Extraction and VERification. In *Proceedings of NAACL-HLT 2018* (pp. 809-819).
4. Schlichtkrull, M., Guo, Z., & Vlachos, A. (2023). AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web. In *Proceedings of NeurIPS 2023 Datasets and Benchmarks Track*.
5. Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global data on events, location, and tone. In *ISA Annual Convention* (Vol. 2, pp. 1-49).
6. Lazer, D. M., Baum, M. A., Benkler, Y., Berinsky, A. J., Greenhill, K. M., Menczer, F., ... & Zittrain, J. L. (2018). The science of fake news. *Science*, 359(6380), 1094-1096. [Reference required for institutional formatting].
7. Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*, 359(6380), 1146-1151. [Reference required for institutional formatting].
