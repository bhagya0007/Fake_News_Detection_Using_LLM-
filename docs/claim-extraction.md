# Claim Extraction Service (Step 11)

## Overview & Purpose

The **Claim Extraction Service** (`backend/app/services/claim_extractor.py`) is the initial analytical stage of the hybrid multi-signal pipeline. Its sole responsibility is to decompose complex unstructured text (news headlines, single statements, press releases, or paragraphs) into at most **5 atomic, checkable factual claims** suitable for subsequent evidence retrieval and verification.

```text
User Text
    │
    ▼
Validation & Conservative Cleaning (backend/app/services/text_cleaner.py)
    │
    ▼
ClaimExtractor (backend/app/services/claim_extractor.py)
    │
    ▼
LLMClient (backend/app/ai/llm_client.py)
    │
    ▼
Configured Provider (MockLLMProvider or live provider via provider_factory.py)
    │
    ▼
Validated ClaimExtractionResult (backend/app/schemas.py)
    │
    ▼
Maximum 5 Atomic Checkable Claims
```

> [!IMPORTANT]
> **Purely Descriptive Component:** The Claim Extraction layer extracts assertions exactly as presented. It does **NOT** verify whether claims are true or false, does **NOT** assign credibility scores, does **NOT** evaluate authenticity, and does **NOT** emit fake news verdicts.

---

## What Counts as a Checkable Claim?

A **checkable factual claim** is an atomic or reasonably self-contained proposition whose truth value can empirically be verified or falsified using external records, data, documentation, or authoritative sources.

### Examples of Checkable Factual Claims:
- *"India's GDP grew by 7.8% last quarter."* (Empirically checkable statistical assertion)
- *"The health ministry announced a new vaccination programme on Monday."* (Empirically checkable event / announcement)
- *"Company X acquired Company Y for $2 billion."* (Empirically checkable commercial transaction)

### Examples of Non-Checkable Content (Excluded or Marked `checkable=False`):
- *"I think this policy is terrible."* (Subjective value judgment / opinion)
- *"This movie is amazing and deeply inspiring."* (Aesthetic commentary)
- *"You should vote for this person."* (Normative recommendation / call to action)
- *"Can you believe what happened yesterday?"* (Rhetorical question)

Pure opinions and rhetorical filler are excluded from the extracted claim list so as not to consume the 5-claim quota.

---

## Core Extraction Principles

### 1. Maximum 5 Claims
The service enforces a strict upper limit of **5 claims** in two redundant layers:
- **Prompt Instruction:** The system prompt explicitly commands the model to extract at most 5 central factual assertions.
- **Pydantic Contract & Service Validation:** [`ClaimExtractionResult.claims`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/schemas.py#L182-L210) specifies `max_length=5` with a dedicated field validator. Any provider output exceeding 5 items is rejected with an [`LLMResponseValidationError`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/ai/exceptions.py#L46-L50).

### 2. Zero External Knowledge & No Claim Correction
The extraction service must remain completely faithful to the input text:
- If the source states *"The Moon is made of green cheese"*, the extracted claim is *"The Moon is made of green cheese"*.
- The extractor must **never** use pre-trained world knowledge to "correct" false-looking claims to *"The Moon is not made of green cheese"*. Claim extraction is not fact-checking.

### 3. Attribution Preservation (Anti-Hallucination)
When a factual assertion is framed as an attribution or quote, the attribution must be preserved:
- Source: *"The minister stated that unemployment fell to 4%."*
- Correct Extraction: *"The minister stated that unemployment fell to 4%."*
- Prohibited Extraction: *"Unemployment fell to 4%."* (Unqualified assertion hallucinations are strictly prohibited because the source only attests that the minister uttered the claim).

### 4. Preservation of Numbers, Dates, and Named Entities
Conservative text normalization via [`clean_text()`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/text_cleaner.py#L21-L62) collapses formatting whitespace while preserving:
- Proper nouns, organization titles, and geographic locations (e.g. `NASA`, `Jaipur`, `Artemis I`).
- Percentages, decimals, and monetary values (e.g. `27.5%`, `$4.2 billion`).
- Temporal markers and calendar dates (e.g. `October 24, 1945`, `2027`).

### 5. Short Input Handling
When the input is already a single short factual claim (e.g. *"India's population exceeded 1.4 billion"*), the service returns that single claim directly without decomposing it into fragments.

### 6. Deterministic Deduplication
If the LLM returns identical duplicate claims after conservative normalization, the service deduplicates them while preserving the original order of first occurrence.

---

## Structured Output Schema

Extracted claims conform strictly to the Pydantic models in [`backend/app/schemas.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/schemas.py):

```python
class ClaimType(str, Enum):
    numeric = "numeric"
    event = "event"
    attribution = "attribution"
    causal = "causal"
    existence = "existence"
    other = "other"

class Claim(BaseModel):
    id: Optional[int] = None
    claim_index: Optional[int] = None
    text: str
    checkable: bool = True
    claim_type: Optional[ClaimType] = None
    source_span: Optional[str] = None

class ClaimExtractionResult(BaseModel):
    claims: list[Claim]  # max 5
    source_text_length: int
    truncated: bool = False
```

### Prohibited Fields:
- No `verdict`, `label`, `is_fake`, or `truth_value` field.
- No `chain_of_thought`, `cot`, `reasoning`, or `thinking` field.

---

## Prompt Versioning

Prompts are stored independently from business logic in [`backend/app/ai/prompts/`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/ai/prompts):
- **Current Version:** `claim_extraction_v1` ([`backend/app/ai/prompts/claim_extraction_v1.txt`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/ai/prompts/claim_extraction_v1.txt))
- [`ClaimExtractor`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/claim_extractor.py#L27-L148) stores and exposes `prompt_version = "claim_extraction_v1"`. Any future prompt tuning must be explicitly versioned (e.g., `claim_extraction_v2.txt`).

---

## Mock vs. Live Provider Behavior

- **Automated Tests:** All unit and integration tests run entirely against [`MockLLMProvider`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/ai/providers/mock_provider.py), completely offline with zero network calls and zero credit consumption.
- **Demonstration Script (`scripts/demo_claim_extractor.py`):**
  - **Default execution:** Runs in `MODE: MOCK / OFFLINE` over 5 distinct text archetypes (single claim, multi-claim paragraph, opinion-only, attributed statement, numerical claim).
  - **Live execution (`--live`):** Requires explicit configuration of `LLM_PROVIDER` and `LLM_API_KEY` in `.env`. If unconfigured, exits with a clear non-error diagnostic.

---

## Known Limitations & Boundaries

1. **Descriptive, Not Verificational:** This component cannot determine whether a claim is true or false.
2. **Context Independence:** For complex journalistic narratives, extraction separates claims into atomic units; relationships between claims will be resolved in the fusion layer.
3. **Upper Bound:** Texts containing more than 5 distinct factual claims are capped to the 5 most central assertions.
