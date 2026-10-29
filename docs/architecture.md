# Architecture Plan & System Design

> **Current status: Core pipeline, persistence, FastAPI backend, and Streamlit frontend are FULLY IMPLEMENTED (Steps 1–18 complete).**

This document outlines the planned end-to-end architecture and pipeline for the Fake News Detection system.

---

## Planned Data Flow

```text
User Text
    ↓
Input Validation / Cleaning
    ↓
Claim Extraction
    ↓
┌──────────────────────────────┐
│ Red-Flag Detector            │
│ Classical ML Baseline        │
│ Evidence Retrieval           │
└──────────────────────────────┘
    ↓
Structured LLM Assessment
    ↓
Fusion + Abstention
    ↓
Final Assessment
    ↓
API / Database / Streamlit UI
```

---

## Component Descriptions (Planned)

1. **Input Validation / Cleaning:** Validates raw text length, formatting, and language prior to processing.
2. **Claim Extraction:** Extracts key checkable propositions from the input text using prompt-based LLM claim isolation.
3. **Multi-Signal Analysis:**
   - **Red-Flag Detector:** Rule-based and heuristic detection of linguistic signals (sensationalism, suspicious phrasing, formatting anomalies).
   - **Classical ML Baseline:** Fast TF-IDF + Logistic Regression classification trained on verified datasets.
   - **Evidence Retrieval:** Optional retrieval module to cross-reference claims against reference knowledge/corpora.
4. **Structured LLM Assessment:** Evaluates retrieved evidence and claims with chain-of-thought verification to output structured analysis.
5. **Fusion + Abstention:** Combines probabilities, heuristics, and LLM reasoning. If uncertainty is above threshold or contradictory, yields abstention ("Unverifiable").
6. **Final Assessment & Delivery:** Emits one of four verdicts (`Likely Reliable`, `Misleading`, `Likely Fake`, `Unverifiable`) to the FastAPI backend, persisted in SQLite, and rendered in Streamlit.

---

## Text Preparation

The project implements a conservative, signal-preserving text preprocessing policy:
- **Raw Statement Preservation:** The original `statement` is preserved untouched across all data pipelines.
- **Baseline Feature:** The normalized `clean_statement` field is utilized for classical ML baseline modeling.
- **Minimal Normalization:** Preprocessing strictly standardizes erratic whitespace, newlines, tabs, and invisible control characters into single spaces, stripping edge padding.
- **Signal Retention:** Punctuation (e.g., `!!!`, `???`), numbers/percentages (e.g., `27.5%`, `2024`), capitalization (e.g., `BREAKING`, `NASA`), and URLs are strictly preserved.
- **Avoidance of Aggressive NLP Operations:** Lowercasing, stopword removal, stemming, and lemmatization are intentionally omitted at this stage because stylistic, sensational, and casing signals are critical for downstream red-flag detection and LLM assessment. (Standard lowercasing will be handled modularly within the TF-IDF vectorizer during feature extraction).

---

## Implemented Classical ML Baseline Pipeline

The classical machine learning baseline follows this verified pipeline:

```text
clean_statement
    ↓
TfidfVectorizer (lowercase=True, ngrams=(1,2), min_df=2, max_df=0.95, max_features=50,000, sublinear_tf=True)
    ↓
LogisticRegression (solver=lbfgs, C=1.0, max_iter=1000, random_state=42)
    ↓
Three-Class Prediction & Class Probability Estimates
```

### Supported Output Labels
The baseline classifier exclusively outputs one of three project labels:
- `likely_reliable`
- `misleading`
- `likely_fake`

### Abstention Handling Note
The classical baseline model does **not** emit `unverifiable`. The `unverifiable` verdict belongs exclusively to the planned hybrid fusion and abstention layer when multi-signal evidence or confidence is insufficient.

---

## Red-Flag / Stylometric Layer (Step 8)

The red-flag detector (`backend/app/services/red_flags.py`) provides interpretable, rule-based feature extraction and presentation-cue detection:

```text
Statement Text
      ↓
[Regex & Stylometric Analysis]
      ↓
┌──────────────────────────────────────────────┐
│ Features: word count, uppercase ratio,       │
│           punct count, stats, URLs, quotes   │
│ Flags   : excessive_caps, sensational,       │
│           urgency, stats, attribution        │
│ Summary : flag counts by severity (H/M/L)    │
└──────────────────────────────────────────────┘
      ↓
Inputs to Future Hybrid Fusion Layer
(Zero truth verdict emitted at this stage)
```

### Key Properties:
- **Deterministic:** Pure Python and regular expression logic without stochastic components.
- **Interpretable:** Every emitted flag provides a transparent explanation, matched snippet, and linguistic severity level (`low`, `medium`, `high`).
- **No Training:** Pure rule-based architecture requiring no gradient updates or supervised label fitting.
- **Linguistic Focus:** Detects presentation and style patterns (sensational buzzwords, urgency calls, capitalization anomalies, missing attribution cues).
- **Non-Decisional:** Explicitly does **not** decide whether a claim is true or false. A sensational headline can be factually true, and a fabricated claim can be written in sober academic language.
- **Fusion Role:** In later steps, the red-flag summary and feature vector will contribute an auxiliary signal to the Bayesian/hybrid fusion system alongside classical ML probabilities and structured LLM evidence reasoning.

---

## Structured Data Contracts (Step 9)

To guarantee type safety, reliable inter-component communication, and clean API serialization, strict Pydantic v2 schemas and enums are defined in [`backend/app/schemas.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/schemas.py):

```text
Inbound Text
    ↓
AnalysisRequest (sanitizes & enforces <=20,000 char limit)
    ↓
┌────────────────────────────────────────────────────────┐
│ Pipeline Components (Validated Contracts)              │
│   • RedFlagResult (15 features, warning flags, summary)│
│   • MLAssessment (MLLabel domain, probabilities)       │
│   • ClaimResult & EvidenceItem (claim & source links)  │
│   • LLMAssessment (structured audit, no hidden CoT)    │
└────────────────────────────────────────────────────────┘
    ↓
FinalAssessment (VerdictLabel domain, support_score, abstention consistency)
    ↓
AnalysisResponse (comprehensive payload with subsystem statuses)
```

### Architectural Distinctions:
1. **Separation of Label Domains:**
   - **Classical ML (`MLLabel`):** Strictly restricted to `likely_reliable`, `misleading`, and `likely_fake`. The classical model cannot judge its own epistemic uncertainty.
   - **Hybrid System (`VerdictLabel`):** Adds `unverifiable` strictly for the fusion layer when corroborating evidence is unavailable or contradictory.
2. **Abstention Consistency:**
   - A verdict of `unverifiable` strictly requires `abstained = True`.
   - Conversely, any definitive verdict (`likely_reliable`, `misleading`, `likely_fake`) strictly requires `abstained = False`.
3. **Probability & Confidence Clarity:**
   - Softmax outputs in `ClassProbabilities` and internal `support_score` are decision-support values; they are not assumed to be calibrated objective truth probabilities.
4. **Structured LLM Assessment Without Free-Form Parsing:**
   - `LLMAssessment` enforces structured categorization (`SourceAttribution`, `EvidenceSupport`, `InternalConsistency`, concise `short_reason`).
   - Unrestricted chain-of-thought, hidden reasoning steps, or private reasoning fields are strictly prohibited from being persisted or transferred in schemas.

---

## LLM Provider Adapter

All LLM operations in the project are encapsulated behind a single, provider-agnostic adapter layer implemented in [`backend/app/ai/`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/ai/):

```text
Business Service (e.g. claim extraction, reasoning)
    ↓
LLMClient (backend/app/ai/llm_client.py)
    ↓
Provider Interface (backend/app/ai/provider_base.py)
    ↓
Configured LLM Provider (backend/app/ai/providers/: MockLLMProvider or GeminiProvider)
    ↓
Validated Pydantic Response (strongly-typed object returned to service)
```

### Key Principles:
1. **Zero Direct SDK Access:** No business service (such as future claim extractors or reasoners) imports or interacts directly with external LLM provider SDKs (OpenAI, Gemini, Anthropic). All interactions route exclusively through `LLMClient.generate_structured()`.
2. **Strict Schema Validation:** Every LLM response is parsed and validated against an explicit Pydantic schema before entering the application domain. Arbitrary dictionaries or free-form strings are never passed to business logic.
3. **Provider Agnosticism:** Downstream components specify what structured data contract they expect, remaining completely decoupled from model providers, endpoints, and authentication details.
4. **Resilience & Safety:** The adapter centrally enforces strict timeouts, non-auth transient retries, telemetry tracking, and credential scrubbing so API keys never leak into logs, exceptions, or metadata.

---

## Claim Extraction Layer

The Claim Extraction Layer ([`backend/app/services/claim_extractor.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/claim_extractor.py)) isolates checkable factual propositions from input statements or articles:

```text
User Text
    ↓
Validation
    ↓
ClaimExtractor
    ↓
LLMClient
    ↓
GeminiProvider / MockProvider
    ↓
Validated ClaimExtractionResult
    ↓
Maximum 5 claims
```

### Architectural Principles:
1. **Extraction Only, Never Verification:** This layer extracts claims only. It does **not** verify them, judge truth or falsehood, compute credibility scores, or emit fake-news verdicts.
2. **Strict Quota:** At most 5 atomic checkable propositions are returned per analyzed text.
3. **Preservation Fidelity:** Exact names, quantities, percentages, dates, and speaker attributions are preserved without modification or external knowledge augmentation.

---

## Structured LLM Reasoning Layer (Step 12)

The Structured LLM Reasoning Layer ([`backend/app/services/llm_reasoner.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/llm_reasoner.py)) performs an evidence-bound evaluation of extracted claims against explicitly provided evidence:

```text
Extracted Claims
      ↓
LLM Reasoner
      +
Supplied Evidence
      ↓
Structured LLMAssessment
```

### Architectural Principles:
1. **Individual Signal, NOT the Final Hybrid System Verdict:** `LLMAssessment` is an isolated reasoning signal produced by evaluating claims against supplied evidence. It is **not** the final hybrid system verdict. The final classification and abstention decision belongs exclusively to the downstream Hybrid Fusion layer (Step 14), which integrates this assessment with classical ML probabilities and deterministic red flags.
2. **Evidence-Bound Reasoning:** The reasoner verifies claims exclusively against supplied reference evidence. It never queries the web or invents facts.
3. **Pretrained Memory is NOT Evidence:** Pretrained model memory cannot verify claims. Factual claims without supplied evidence receive `verdict = "unverifiable"` and `evidence_support = "unavailable"`.
4. **Orthogonal Rubric Dimensions:** Evaluates source attribution (`strong`, `partial`, `weak`, `none`, `unknown`), evidence support (`supported`, `contradicted`, `partial`, `unavailable`, `not_applicable`), internal consistency (`consistent`, `questionable`, `contradictory`, `unknown`), rhetorical cues, and unsupported claims.
5. **Zero Hidden Chain-of-Thought:** Pure structured rubric outputs conforming strictly to `LLMAssessment`.

---

## Local Evidence Knowledge Base & Semantic Retrieval Layer (Step 13)

The Local Evidence Retrieval Layer ([`backend/app/services/evidence_retriever.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/evidence_retriever.py)) grounds reasoning by retrieving reference documents for checkable claims:

```text
Extracted Claim
      ↓
EvidenceRetriever
      ↓
Local Evidence KB (JSONL + Dense / Sparse Index)
      ↓
Top-k Relevant Evidence (Filtered by Threshold)
      ↓
LLM Reasoner
```

### Architectural Principles:
1. **Semantic Similarity ≠ Factual Support:** High semantic similarity indicates topical relevance only; it does **not** imply that the evidence supports or verifies the claim. Stance (`supports`, `contradicts`, `related`, `unknown`) is strictly preserved and never inferred from cosine similarity.
2. **Local & Traceable:** Searches strictly within the local validated KB (`data/evidence/evidence_kb.jsonl`). Zero runtime web browsing, scraping, or hallucinated URLs.
3. **No-Sufficient-Evidence Policy:** If no evidence item clears the similarity threshold (`EVIDENCE_SIMILARITY_THRESHOLD`), returns `items = []` and `no_sufficient_evidence = True`. The retriever never forces a poor match.
4. **Representation Backends:** Supports semantic dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and scikit-learn lexical fallback (`tfidf_lexical`).

---

## Multi-Signal Decision Fusion & Abstention Engine (Step 14)

The Hybrid Decision Fusion Layer ([`backend/app/services/fusion.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/fusion.py)) integrates all upstream subsystem signals into an authoritative, auditable, and uncalibrated heuristic `FinalAssessment`:

```text
ML Signal ─────────────┐
Red Flags ─────────────┤
Evidence Retrieval ────┼→ Fusion / Abstention Engine → FinalAssessment
LLM Assessment ────────┘
```

### Architectural Principles:
1. **Evidence availability gates factual verdicts:** For any checkable factual claim, usable provenance-bearing, non-synthetic evidence is a mandatory prerequisite. If no usable provenance-bearing evidence is retrieved (`no_sufficient_evidence=True` or zero usable non-synthetic items), the system **must abstain** (`verdict = "unverifiable"`, `abstained = True`). Classical ML and red flags can never override the absence of evidence. Note that provenance alone does not prove factual correctness, but its absence mandates abstention.
2. **Not a Majority Vote:** Fusion is a hierarchical rule engine, not an unweighted ensemble. Classical ML (Macro F1 = 0.4345) and red-flag heuristics are secondary indicators that cannot outvote grounded factual evidence.
3. **Red Flags Boundaries:** Stylometric cues (sensationalism, excessive capitalization, urgency phrases) reflect presentation style only. They can explain tone and modulate decision support score, but can never independently classify a claim as fake or misleading.
4. **Synthetic Evidence Quarantine:** Synthetic fixtures (`is_synthetic=True`) are quarantined by default (`allow_synthetic_evidence=False`). Only real-world verified records can substantiate production fact checks.
5. **Decision-Support Score Semantics:** `support_score` is an uncalibrated heuristic measure of decision-support strength (0.0 to 1.0) given the agreement of multiple independent signals, never an objective calibrated probability of truth. When abstaining, `support_score` is strictly 0.0.

---

## End-to-End Hybrid Engine Orchestration (Step 15)

The `HybridAnalysisEngine` ([`backend/app/services/hybrid_engine.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/hybrid_engine.py)) unifies all independent pipeline stages into a single entry point:

```text
User Text
   ↓
Validation/Cleaning
   ↓
Claim Extraction
   ├────────────→ Classical ML
   ├────────────→ Red Flags
   ↓
Evidence Retrieval
   ↓
LLM Reasoner (Short-circuited if no usable evidence)
   ↓
Fusion / Abstention
   ↓
AnalysisResponse
```

### Architectural Principles & Core Roles:
1. **Classical ML and Red Flags are Signals:** The classical model and red-flag detector produce stylistic and linguistic feature cues. They do not constitute factual proof.
2. **Evidence + Reasoner Provide Grounding:** The local evidence knowledge base and structured reasoner provide verifiable grounding.
3. **Fusion Alone Creates the Final Verdict:** The orchestrator delegates decision-making entirely to `fusion.py`. No component or coordinator overrides the fusion engine.
4. **Quota-Aware Evidence Gate (Step 15.5):** If no usable non-synthetic evidence is retrieved for checkable claims, the engine automatically short-circuits the second LLM call (LLM reasoner). The status is marked `skipped`, saving provider quota and latency without changing the final verdict semantics (`unverifiable`).
5. **Resilience & Graceful Degradation:** Provider rate limits (429 quota exhaustion), request timeouts, and retrieval unavailability safely degrade to `unverifiable` abstention without crashing the application.

> [!IMPORTANT]
> **Prominent Current Limitation:** The current production evidence knowledge base contains zero records. Therefore, production-mode factual claims will generally abstain as **Unverifiable** until provenance-bearing evidence is added.

---

## SQLite Persistence & Audit Layer (Step 16)

The persistence layer ([`backend/app/db/`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/db/)) introduces audit-grade, local-only relational storage for completed analyses and user feedback without coupling the pure analysis pipeline to database state:

```text
HybridAnalysisEngine
        ↓ (AnalysisResponse)
  AnalysisService
        ↓
 AnalysisRepository
        ↓
      SQLite (data/app/fake_news_analysis.db)
```

### Key Architectural Principles:
1. **Database Stores Outputs, Never Influences Verdicts:** The database is strictly a downstream audit and query sink. Verdict generation occurs independently within `HybridAnalysisEngine` and is never informed by previously stored rows.
2. **Deterministic Failure Isolation:** In `AnalysisService.analyze_and_save()`, multi-signal analysis is executed first. If SQLite writing fails, a controlled `DatabaseError` is raised, but the computed verdict and decision-support score are never modified or altered.
3. **Structured JSON Audit Reconstruction:** The full validated `AnalysisResponse` is persisted as structured JSON in the `analyses` table, enabling instantaneous round-trip reconstruction without recomputing classical ML, vector embeddings, or LLM reasoning.
4. **User Feedback Segregation:** User feedback records are collected purely for qualitative review and post-hoc error analysis. Feedback is strictly isolated and never used for automated model retraining, prompt tuning, or fusion weight adjustment.
5. **Relational Cascade:** Foreign key enforcement (`PRAGMA foreign_keys = ON;`) ensures that deleting an analysis record automatically cascades and purges all attached feedback items.

---

## FastAPI REST API Layer (Step 17)

The FastAPI delivery layer ([`backend/app/api/`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/api/)) provides a modern, high-performance, asynchronous REST service wrapping the existing persistence and hybrid engine services:

```text
HTTP Client / Streamlit UI (Step 18)
                │
                ▼
      FastAPI Application (backend/app/api/app.py)
        ├── Health Router   (/healthz, /api/v1/health)
        ├── Analysis Router (/api/v1/analyze, /api/v1/analyses)
        └── Feedback Router (/api/v1/analyses/{id}/feedback)
                │
                ▼
       AnalysisService (backend/app/services/analysis_service.py)
          ┌──────┴───────────────────────────────────────┐
          ▼                                              ▼
HybridAnalysisEngine (In-Memory)        AnalysisRepository (SQLite DB)
```

### Architectural Principles & API Guarantees:
1. **Thin Delivery Layer:** The API strictly handles request validation, dependency injection, and HTTP response serialization. It contains zero duplicated ML inference, retrieval logic, or fusion math.
2. **Lifespan State Dependency Injection:** Component singletons (`AnalysisRepository`, `HybridAnalysisEngine`, `AnalysisService`) are resolved during application startup and bound to `app.state`, enabling zero-cost test isolation via custom constructor injections.
3. **Non-Blocking Threadpool Execution:** Downstream synchronous operations (classical ML scikit-learn models, sentence-transformers, SQLite operations) run automatically inside Starlette's threadpool worker executors, keeping the ASGI event loop responsive.
4. **Safe Abstention Returns HTTP 200:** When the hybrid engine determines that evidence is absent or contradictory (`verdict="unverifiable"`, `abstained=True`, `support_score=0.0`), the API responds with `200 OK` rather than an error. Abstention is an expected, trustworthy domain outcome.
5. **Constant-Time Health Checks:** `/healthz` and `/api/v1/health` return instantaneous static status indicators without triggering LLM calls, classical model inference, or database transactions.
6. **Strict Production Evidence Quarantine:** The production API exposes no synthetic evidence toggles (`allow_synthetic_evidence`). Synthetic test fixtures are strictly quarantined from public API endpoints and OpenAPI documentation.
7. **Sanitized Error Shielding:** Database connectivity issues, schema violations, and unhandled server errors are mapped to RFC 7807-style error responses (`503 Service Unavailable`, `422 Unprocessable Entity`, `500 Internal Server Error`) with zero leakage of SQL statements, stack traces, or credentials.

---

## Streamlit Frontend Layer (Step 18) — IMPLEMENTED

The Streamlit frontend ([`frontend/`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/frontend/)) provides an interactive, accessible user interface for the verification pipeline:

```text
Streamlit Frontend (frontend/app.py)
              │
              │ HTTP (APIClient via httpx)
              ▼
FastAPI Backend (backend/app/api/app.py)
              │
              ▼
       AnalysisService (backend/app/services/analysis_service.py)
        ┌─────┴────────────────────────────────────────┐
        ▼                                              ▼
HybridAnalysisEngine (In-Memory)        AnalysisRepository (SQLite DB)
```

### Architectural Guarantees & Frontend Boundaries:
1. **Pure API Client:** The frontend operates exclusively as an HTTP consumer of the FastAPI backend. It contains zero local ML inference, zero direct database access, zero direct LLM SDK calls, and zero duplicated verdict calculation logic.
2. **Strict Layer Isolation:** Frontend modules strictly never import `scikit-learn`, `joblib`, `google.genai`, `sqlite3`, or internal pipeline orchestrators.
3. **Consistent Validation Bounds:** Accepts up to 20,000 characters matching backend and engine validation boundaries, with live client-side character counting.
4. **Heuristic Score Semantics:** Explicitly labels `support_score` as "Decision-support strength" (0.00 to 1.00) with prominent tooltips clarifying it is an uncalibrated heuristic measure, never a mathematical probability of truth.
5. **Auditable History Inspection:** Historical analyses are browsed and reopened directly from SQLite persistence without recomputing ML models or LLMs.
6. **Graceful Offline Degradation:** If the FastAPI backend is offline, the UI alerts the user without crashing and disables analysis actions.

---

## Version 2 Branch: Single-Provider Groq Runtime (V2.1)

In Version 2.1 (`v2-groq` branch), the active LLM runtime transitions from Gemini to Groq as the sole active provider.

### Architecture Overview:
```text
HybridAnalysisEngine
         │
         ▼
    LLMReasoner
         │
         ▼
     LLMClient (Provider-Neutral)
         │
         ▼
    GroqProvider (backend/app/ai/providers/groq_provider.py)
         │  (Strict JSON Schema, Model: openai/gpt-oss-20b)
         ▼
    Groq Cloud API (console.groq.com)
```

### Version 2 Architectural Guarantees:
1. **Single Active Provider:** Groq is the sole active LLM provider. There is no multi-provider fallback chain, no fallback to Gemini, and no provider rotation.
2. **Explicit Disallowal of Gemini Runtime:** `provider_factory` rejects `provider="gemini"` with `LLMConfigurationError("V2 runtime is configured for Groq-only operation.")`.
3. **Strict JSON Schema Normalization:** All Pydantic response models pass through `_normalize_schema_for_strict()` enforcing `additionalProperties: False` and recursive `required` property listings to ensure 100% adherence to Groq strict structured output constraints.
4. **Secret Protection & Redaction:** `GROQ_API_KEY` is never written to logs, database records, API outputs, or exception strings.
5. **Zero-Network Automated Tests:** All automated unit and integration tests execute with zero network socket operations via mock clients and pytest socket blockers.

---

## Version 2 Branch: Real-World News Evidence Discovery (V2.2)

In Version 2.2, the pipeline incorporates **real-world, web-based, provenance-bearing news evidence discovery** using public GDELT endpoints.

### Architecture Overview:
```text
Checkable Factual Claim
         │
         ▼
EvidenceQueryBuilder (Deterministic NLP, zero LLM, strips operators, caps at 150 chars / 8 terms)
         │
         ▼
GDELTClient (Public HTTP, polite pacing >=3.2s, in-memory TTL cache, zero secret API keys)
   ├── Primary: GDELT Context 2.0 API (context snippets, sentence excerpts)
   └── Fallback: GDELT DOC 2.0 API (headline annotations)
         │
         ▼
LiveEvidenceRetriever (Canonical URL deduplication, domain diversity counting)
         │
         ▼
MiniLM Semantic Reranking (all-MiniLM-L6-v2 cosine similarity)
         │
         ▼
Candidate Evidence Items (stance="unknown", is_synthetic=False, auditable URL + domain)
         │
         ▼
HybridAnalysisEngine (records live_retrieval_ms, deduplicates with local KB, passes to fusion)
```

### Version 2.2 Epistemic and Security Invariants:
1. **Discovery Source, Not Truth Authority:** GDELT results are strictly treated as *candidate evidence*. They are never assumed to be verified facts or authoritative truth.
2. **Stance Isolation:** In V2.2, all live evidence enters the pipeline with `stance = EvidenceStance.unknown`. Claim-evidence stance reasoning is deferred to V2.3.
3. **Single Secret Maintained:** Zero additional API keys are required or supported. `GROQ_API_KEY` remains the sole secret in the project.
4. **No Web Scraping:** Raw web pages are never scraped or parsed; only structured snippet and title metadata returned by GDELT is utilized.
5. **Deterministic Fusion Safety:** Because stances are unknown, the frozen fusion engine continues to safely abstain (`unverifiable`) on external claims lacking local ground truth.
6. **Full Test Isolation:** Pytest suites mock all HTTP calls and prevent network socket creation, ensuring 100% offline test reproducibility.

---

## Version 2 Branch: Claim ↔ Evidence Stance Reasoning & Deterministic Consensus (V2.3)

In Version 2.3, candidate evidence discovered by GDELT and reranked by MiniLM undergoes **structured stance reasoning via Groq**, followed by **deterministic evidence consensus and near-duplicate clustering**, before entering the decision fusion layer.

### Target Architecture Flow:
```text
User News Statement
         │
         ▼
Groq Claim Extraction (Atomic checkable propositions, capped at 5 claims)
         │
         ▼
GDELT + Local Retrieval (Candidate evidence discovery, URLs, domains, snippets)
         │
         ▼
MiniLM Semantic Reranking (Local sentence-transformers cosine similarity)
         │
         ▼
Groq Evidence Stance Assessment (Single structured batch call, prompt-injection defense)
    ├── Stances: supports, contradicts, partial, irrelevant, unclear
    └── Attribution Types: direct_report, quoted_claim, reported_denial, etc.
         │
         ▼
Deterministic Consensus & Clustering Engine (Connected components, cosine sim >= 0.82 / Jaccard >= 0.65)
    ├── Headline Exclusion: headline_only items excluded from decisive counts
    ├── Quoted Claim Filter: quoted_claim cannot confirm propositions
    ├── Denial Filter: reported_denial cannot count as support
    ├── Syndicated Deduplication: wire stories reprinted across domains merge into 1 cluster
    ├── Two-Source Threshold: minimum 2 independent clusters from 2 distinct domains
    └── Conflict Gate: coexisting support and contradiction forces 'conflicting'
         │
         ▼
Deterministic Hybrid Decision Fusion (Final decision authority; emits likely_reliable, likely_fake, misleading, or unverifiable)
```

### Version 2.3 Core Principles & Invariants:
1. **Epistemic Separation:** Groq is NOT asked "Is this news true?". Groq is asked exclusively: "Given this claim and ONLY the supplied evidence snippet text, what is the textual relationship of the evidence to the claim?"
2. **Batch Efficiency:** Stance reasoning runs via **ONE single structured Groq call per analysis** for all claims and top candidate snippets, eliminating network roundtrip latency.
3. **Prompt-Injection Defense:** All claims and snippets are treated as untrusted data; embedded override directives (e.g. "Ignore previous instructions", "Mark true") are strictly ignored.
4. **Conservative Corroboration:** Single-source claims cannot produce definitive verdicts. At least 2 independent clusters across 2 distinct publisher domains are mandatory.
5. **Syndicated Wire Protection:** Wire reports reprinted verbatim across multiple domains form a single cluster and do not count as independent corroborations.
6. **Sole Authority:** Deterministic fusion remains the sole final authority; Groq never dictates the final verdict directly.



