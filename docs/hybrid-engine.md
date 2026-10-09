# End-to-End Hybrid Engine Orchestration (Step 15)

This document details the architecture, execution pipeline, graceful degradation policies, and operational characteristics of the unified `HybridAnalysisEngine` (`backend/app/services/hybrid_engine.py`).

---

## 1. Overview & Architectural Role

The `HybridAnalysisEngine` acts as the single central controller and coordinator for the fake news detection platform. It accepts arbitrary user-submitted statements or article excerpts and executes a deterministic, multi-signal pipeline combining classical machine learning, stylometric analysis, semantic evidence retrieval, structured LLM reasoning, and epistemic fusion.

```text
User Text
   ↓
Validation / Cleaning
   ↓
Claim Extraction
   ├────────────→ Classical ML Baseline
   ├────────────→ Stylometric Red Flags
   ↓
Evidence Retrieval
   ↓
Deduplication & Grounding
   ↓
Structured LLM Reasoner
   ↓
Deterministic Fusion & Abstention
   ↓
Final AnalysisResponse
```

### Core Architectural Distinctions:
1. **Classical ML and Red Flags are Signals:** Text classification patterns and linguistic heuristics serve as auxiliary diagnostic cues. They cannot establish truth or falsehood independently.
2. **Evidence and Reasoner Provide Grounding:** Factual verification requires non-synthetic provenance-bearing reference documents. The LLM reasoner evaluates claims strictly against supplied evidence.
3. **Fusion Alone Creates the Final Verdict:** The engine coordinates and passes signals, but `fusion.py` remains the single authoritative decision engine. The orchestrator never overrides fusion.

---

## 2. Pipeline Execution Sequence

The engine enforces a strict 10-step execution order:

| Step | Component | Method / Source | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Input Validation** | `_validate_input` | Enforces non-empty string, type safety, and maximum length bound ($\le 20,000$ characters). Rejects silently truncated inputs. |
| **2** | **Conservative Cleaning** | `clean_text` | Normalizes whitespace noise while strictly preserving capitalization, punctuation clusters, numbers, and URLs. |
| **3** | **Claim Extraction** | `ClaimExtractor.extract_claims` | Isolates at most 5 atomic checkable factual propositions using prompt-based LLM claim extraction. |
| **4** | **Classical ML Baseline** | `DefaultMLService.predict` | Performs fast inference using the frozen TF-IDF + Logistic Regression model (`models/classical_best_validation.joblib`). |
| **5** | **Red-Flag Analysis** | `DefaultRedFlagService.analyze` | Evaluates raw input text against 10 deterministic regex rule categories (15 stylometric features). |
| **6** | **Evidence Retrieval** | `EvidenceRetriever.retrieve` | Queries the local knowledge base per checkable claim (`Claim.text`) using semantic embeddings or lexical search. |
| **7** | **Evidence Aggregation** | `seen_keys` deduplication | Merges and deduplicates retrieved records by evidence ID across all claims while preserving claim associations in `ClaimResult`. |
| **8** | **LLM Reasoner** | `LLMReasoner.assess` | Evaluates claims and evidence across orthogonal rubric dimensions (source attribution, evidence support, consistency). **Short-circuited if no usable evidence is found.** |
| **9** | **Deterministic Fusion** | `fusion.fuse` | Evaluates 8-tier hierarchical rules, evidence gating, and computes uncalibrated heuristic decision-support score. |
| **10** | **Response Assembly** | `AnalysisResponse` | Compiles verdict, claims, evidence, component statuses, latency breakdown (`PipelineTimings`), and audit justification. |

---

## 3. Quota-Aware Evidence Gate (Step 15.5)

To conserve commercial LLM API quota (e.g., Gemini rate limits) and eliminate pointless network latency, the engine implements a **quota-aware evidence gate**:

```text
Evidence Retrieval
       ↓
Usable Evidence Found?
     /        \
   YES         NO
   /            \
Run LLM      Short-Circuit LLM
Reasoner     (Status: skipped)
   \            /
    ↓          ↓
Deterministic Fusion (verdict = unverifiable)
```

### Key Principles:
1. **Evidence-Bound Semantics:** The LLM reasoner is strictly evidence-bound. If no external evidence is supplied, it is architecturally constrained to emit `unverifiable` (`evidence_support = "unavailable"`).
2. **Quota Preservation:** Invoking an external LLM when evidence is absent adds 5–25 seconds of network latency and consumes API quota without altering the final verdict. Fusion already guarantees that checkable claims with zero evidence abstain as `unverifiable`.
3. **Claim Extraction Still Runs:** The first LLM call (`ClaimExtractor`) is **never** skipped. It is essential to isolate atomic propositions, establish checkability, and formulate retrieval queries.
4. **Transparent Status Reporting:** When short-circuited, `llm_reasoner` records `status = "skipped"` with message `"Skipped because no usable external evidence was available."`.
5. **Standalone Reasoner Unaffected:** The standalone `LLMReasoner` class continues to accept no-evidence inputs for direct testing; the short-circuit optimization resides strictly within the `HybridAnalysisEngine` orchestrator.

---

## 4. Failure Handling & Graceful Degradation

The orchestrator is engineered to be resilient against provider outages, rate limits, and network errors:

| Component | Failure Condition | Component Status | Policy & Degraded Behavior |
| :--- | :--- | :--- | :--- |
| **Claim Extractor** | LLM error or validation error | `failed` | Claims default to `[]`. Factual claims cannot be established; pipeline skips retrieval and reasoner, yielding `unverifiable`. |
| **Classical ML** | Artifact load or inference error | `unavailable` | ML signal is marked `None`. Pipeline proceeds using evidence, reasoner, and red flags. |
| **Red Flags** | Regex or parsing exception | `unavailable` | Red-flag signal is marked `None`. Pipeline proceeds without presentation penalties. |
| **Evidence Retrieval**| Index missing or disk I/O error | `failed` | Evidence treated as unavailable (`items = []`). Pipeline safely abstains under the evidence gate. |
| **LLM Reasoner** | **429 Quota Exceeded / Rate Limit** | `unavailable` | Records `"LLM unavailable: quota exhausted or rate limited"`. Pipeline safely abstains (`unverifiable`). |
| **LLM Reasoner** | **Request Timeout** | `unavailable` | Records `"LLM unavailable: request timed out"`. Pipeline safely abstains (`unverifiable`). |
| **Fusion Engine** | Internal unhandled exception | `failed` | Emergency fallback: safely returns `unverifiable` with `support_score = 0.0` and `abstained = True`. |

---

## 5. Evidentiary Policy & Production Quarantine

### Prominent Current Limitation
> [!IMPORTANT]
> **The current production evidence knowledge base contains zero records.**
> Therefore, production-mode factual claims will generally abstain as **Unverifiable** until non-synthetic, provenance-bearing evidence is added.

### Production vs. Synthetic Demo Behavior
- **Default Production Mode (`allow_synthetic_evidence=False`):** Any retrieved record marked `is_synthetic=True` is quarantined and ignored. If all retrieved records are synthetic, `has_usable_evidence` evaluates to `False`, triggering the LLM short-circuit and safe abstention.
- **Explicit Demo Mode (`allow_synthetic_evidence=True`):** Enabled exclusively via internal dependency injection or `HybridAnalysisEngine.demo(...)` for integration tests and demonstrations. In demo mode, synthetic fixtures are treated as usable, allowing the reasoner to evaluate groundings.
- **Public API Isolation:** `allow_synthetic_evidence` is strictly prohibited from being exposed as a public parameter in `analyze_text(text: str)`.

---

## 6. Support Score & Telemetry

### Support Score Semantics
`support_score` (ranging from $0.0$ to $1.0$) is an **uncalibrated heuristic decision-support score** measuring multi-signal concordance and evidence grounding.
- It is **NOT** a calibrated objective truth probability.
- When the system abstains (`unverifiable`), `support_score` is strictly fixed to $0.0$.
- In the public API and documentation, it is never labeled as `truth_probability` or `confidence_probability`.

### Latency Categories & Timing Metadata
The engine reports non-negative millisecond timers in `PipelineTimings` (`cleaning_ms`, `claim_extraction_ms`, `ml_ms`, `red_flags_ms`, `retrieval_ms`, `llm_ms`, `fusion_ms`, `total_ms`).

**Latency Distinction:**
- **Local Component Latency:** In-process CPU execution (text cleaning $<0.1\text{ ms}$, classical ML inference $\sim 1\text{–}3\text{ ms}$, red-flag regex $<0.2\text{ ms}$, fusion $<0.1\text{ ms}$).
- **Offline / Mock Latency:** Controlled test runs using test doubles complete in $\approx 1\text{–}3\text{ ms}$ total. Mocked latency must **never** be presented as live production latency.
- **Live Provider Latency:** Network round-trips to live commercial providers (e.g. Gemini API) require $\approx 5\text{–}25\text{ seconds}$ per call. By short-circuiting no-evidence reasoner calls, the engine eliminates an entire 10–25s network round-trip.

