# Structured Data Schemas and Contracts Specification (Step 9)

> **Purpose & Scope:**  
> This document specifies the authoritative, strongly-typed Pydantic v2 data contracts implemented in [`backend/app/schemas.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/schemas.py). These contracts establish strict boundaries between pipeline components before integrating the LLM provider, evidence retrieval, and hybrid fusion layers.  
> **Notice:** Component contracts documented below establish interfaces for upcoming subsystems; they do not imply that external LLM or retrieval services are currently active.

---

## 1. Domain Separation of Label Enums

A critical architectural distinction is maintained between classical ML outputs and the full hybrid system:

```text
┌────────────────────────────────────────────────────────┐
│ MLLabel (Classical Baseline Domain)                    │
│   • likely_reliable                                    │
│   • misleading                                         │
│   • likely_fake                                        │
│   (Strictly NO 'unverifiable' - classical models       │
│    cannot evaluate their own epistemic boundaries)     │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ VerdictLabel (Hybrid AI / System Domain)               │
│   • likely_reliable                                    │
│   • misleading                                         │
│   • likely_fake                                        │
│   • unverifiable  (Exclusively for fusion/abstention)  │
└────────────────────────────────────────────────────────┘
```

---

## 2. Core Schema Specifications & Examples

### 1. `AnalysisRequest`
Inbound payload representing user-submitted text for analysis.
- **Validation Rules:**
  - Strips leading and trailing whitespace automatically.
  - Rejects empty strings and whitespace-only text (`ValueError`).
  - Enforces `MAX_ANALYSIS_TEXT_LENGTH = 20,000` characters (configurable).
- **Example:**
```json
{
  "text": "According to the Department of Labor, unemployment fell to 3.8% this quarter."
}
```

---

### 2. `Claim`
Checkable atomic proposition extracted from input text.
- **Fields:**
  - `id`: Optional integer identifier ($\ge 0$).
  - `text`: Non-empty string ($1 \le \text{length} \le 5000$).
  - `checkable`: Boolean indicating whether proposition is empirically verifiable.
  - `source_span`: Optional excerpt string from the original statement.
- **Example:**
```json
{
  "id": 1,
  "text": "Unemployment fell to 3.8% this quarter.",
  "checkable": true,
  "source_span": "unemployment fell to 3.8% this quarter"
}
```

---

### 3. `RedFlagResult`
Output contract validating signals from the Step 8 stylometric analyzer.
- **Sub-Schemas:**
  - `RedFlag`: `type`, `severity` (`low`, `medium`, `high`), `message`, `snippet`, `value` (`int | float | str | None`).
  - `RedFlagFeatures`: The 15 quantitative signals (counts $\ge 0$, ratios $\in [0.0, 1.0]$).
  - `RedFlagSummary`: Aggregated counts by severity and key categorical indicators.
- **Example:**
```json
{
  "flags": [
    {
      "type": "statistical_claim",
      "severity": "low",
      "message": "Explicit percentage assertion detected (informational signal).",
      "snippet": "3.8%",
      "value": 1
    }
  ],
  "features": {
    "character_count": 87,
    "word_count": 13,
    "uppercase_letter_ratio": 0.0435,
    "exclamation_count": 0,
    "question_mark_count": 0,
    "repeated_punctuation_count": 0,
    "percentage_count": 1,
    "statistical_expression_count": 1,
    "url_count": 0,
    "sensational_phrase_count": 0,
    "certainty_phrase_count": 0,
    "urgency_phrase_count": 0,
    "attribution_phrase_count": 1,
    "quote_count": 0,
    "excessive_punctuation_ratio": 0.023
  },
  "summary": {
    "total_flags": 1,
    "high_severity_flags": 0,
    "medium_severity_flags": 0,
    "low_severity_flags": 1,
    "has_attribution": true,
    "has_statistical_claim": true
  }
}
```

---

### 4. `MLAssessment`
Output contract for the frozen classical machine learning model.
- **Validation Rules:**
  - `label`: Must belong to `MLLabel` (`likely_reliable`, `misleading`, `likely_fake`). Strictly rejects `unverifiable`.
  - `probabilities`: `likely_fake`, `likely_reliable`, `misleading` each $\in [0.0, 1.0]$.
  - Probability Sum: Must sum to $1.0 \pm 0.01$ (floating point tolerance).
- **Example:**
```json
{
  "label": "likely_reliable",
  "probabilities": {
    "likely_fake": 0.2702,
    "likely_reliable": 0.4282,
    "misleading": 0.3016
  },
  "model_name": "Frozen Word TF-IDF + Logistic Regression",
  "model_version": "1.0.0"
}
```

---

### 5. `EvidenceItem`
Structured contract for external knowledge retrieved to corroborate or dispute claims.
- **Fields:**
  - `id`: Optional unique identifier.
  - `claim_summary`: Target claim being investigated.
  - `evidence_text`: Verbatim evidence snippet.
  - `source_name`: Authoritative publisher.
  - `similarity_score`: Embedding/lexical similarity $\in [0.0, 1.0]$.
  - `stance`: `supports`, `contradicts`, `related`, `unknown`.
- **Note:** *Semantic similarity does NOT imply stance.* A document with 0.95 similarity score may directly contradict the claim.
- **Example:**
```json
{
  "id": "ev_01",
  "claim_summary": "Q3 unemployment reached 3.8%",
  "evidence_text": "Bureau of Labor Statistics reported non-farm unemployment at 3.8%.",
  "source_name": "U.S. Bureau of Labor Statistics",
  "source_url": "https://www.bls.gov/news.release/empsit.nr0.htm",
  "published_date": "2024-10-04",
  "similarity_score": 0.91,
  "stance": "supports"
}
```

---

### 6. `LLMAssessment`
Structured contract for the upcoming LLM verification module.
- **Strict Anti-Pollution Guarantee:**
  - Contains **NO** `chain_of_thought`, `reasoning_steps`, `private_reasoning`, or `hidden_reasoning` fields.
  - Requires a concise, auditable `short_reason` ($1 \le \text{length} \le 1000$).
  - Bounded lists for `rhetorical_cues` ($\le 20$) and `unsupported_claims` ($\le 20$).
- **Example:**
```json
{
  "label": "likely_reliable",
  "source_attribution": "strong",
  "evidence_support": "supported",
  "internal_consistency": "consistent",
  "rhetorical_cues": [
    "official institutional attribution",
    "specific quantitative rate"
  ],
  "unsupported_claims": [],
  "short_reason": "Statement cites verified Department of Labor quarterly employment metrics."
}
```

---

### 7. `FinalAssessment`
Contract delivering the definitive verdict from the hybrid fusion layer.
- **Semantic Consistency Validation:**
  - If `verdict == "unverifiable"`, `abstained` must be `True`.
  - If `abstained == True`, `verdict` must be `"unverifiable"`.
  - Any mismatch raises an immediate Pydantic `ValidationError`.
- **Support Score:** Documented as an *"Internal decision-support score used by the hybrid fusion layer ($0.0 \le s \le 1.0$), not a calibrated real-world probability."*
- **Example:**
```json
{
  "verdict": "likely_reliable",
  "support_score": 0.82,
  "reason": "Corroborated by classical text baseline and official Department of Labor statistics.",
  "abstained": false,
  "signals_used": [
    "classical_ml",
    "red_flags",
    "structured_llm"
  ]
}
```

---

### 8. `AnalysisResponse`
Top-level payload returned to API callers.
- **Graceful Degradation:** `llm` is `Optional[LLMAssessment] = None`. If the LLM provider experiences latency or outages, the system can return valid responses backed by the classical baseline and red-flag detector, noting degraded status in `component_statuses`.
- **Example:**
```json
{
  "analysis_id": "demo_req_001",
  "input_text": "According to the Department of Labor, unemployment fell to 3.8% this quarter.",
  "verdict": "likely_reliable",
  "support_score": 0.82,
  "claims": [],
  "red_flags": { ... },
  "ml": { ... },
  "llm": { ... },
  "evidence": [],
  "reason": "Corroborated by classical baseline and official statistics.",
  "component_statuses": [
    { "component": "classical_ml", "status": "success" },
    { "component": "red_flags", "status": "success" },
    { "component": "llm", "status": "success" }
  ],
  "created_at": "2026-09-30T16:19:50Z"
}
```

---

### 9. `FeedbackRequest`
Contract for user agreement/disagreement submissions.
- **Fields:**
  - `analysis_id`: Integer or string identifier.
  - `agree`: Boolean agreement.
  - `comment`: Optional feedback string ($\le 2000$ characters).
- **Example:**
```json
{
  "analysis_id": "demo_req_001",
  "agree": true,
  "comment": "Accurate rating corroborated by official sources."
}
```
