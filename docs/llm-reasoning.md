# LLM-Based Evidence-Bound Reasoning Layer

This document details the design, rubric, and operating principles of the structured, evidence-bound LLM reasoning service (`LLMReasoner`) implemented in Step 12.

---

## 1. Purpose and Architecture

The `LLMReasoner` service (`backend/app/services/llm_reasoner.py`) evaluates extracted claims strictly against supplied evidence using an explicit, orthogonal reasoning rubric. It produces a strongly typed `LLMAssessment` Pydantic object that provides audit-ready linguistic and evidentiary signals.

```text
Extracted Claims (ClaimResult)
            ↓
       LLM Reasoner
            +
  Supplied Evidence (Optional)
            ↓
 Structured LLMAssessment
 (verdict, rubric dimensions, unsupported claims, short_reason)
```

### Critical Architectural Boundaries
1. **Not a Final System Verdict:** The `LLMAssessment.verdict` is an **individual signal** produced by the LLM reasoning layer. It is **not** the final hybrid verdict. Downstream fusion (Step 14) combines this assessment with classical ML probabilities and deterministic red flags.
2. **Not a Knowledge Retriever:** The reasoner never queries external search engines, retrieves documents, or interacts with databases. It evaluates *only* what is explicitly passed into `assess()`. Evidence retrieval is handled independently by the retrieval layer (Step 13).
3. **No Retraining or Modifying Prior Layers:** The classical ML model (`models/classical_best_validation.joblib`) remains frozen, the official LIAR test split (`data/raw/test.tsv`) remains untouched, and the red-flag service remains independent.

---

## 2. Core Operating Principles

### A. Strict Evidence-Bound Reasoning
The reasoner evaluates claims **exclusively against the text snippets provided in `evidence`**. It never invents, hallucinates, assumes, or extrapolates facts beyond what the supplied text and evidence state.

### B. Pretrained Memory is NOT Evidence
The model's internal pre-trained memory or world knowledge is **strictly prohibited** from serving as verification evidence.
- If a claim asserts an externally checkable real-world fact (e.g., *"The Moon is made entirely of green cheese"* or *"Company X posted record profits"*), but no evidence is provided, the reasoner **must not** declare it true or false based on what the model learned during pretraining.
- Without supplied evidence, such claims must be assigned `evidence_support = "unavailable"` and `verdict = "unverifiable"`.
- Pretrained linguistic knowledge is used exclusively to evaluate syntax, tone, internal coherence, and attribution cues within the provided text.

### C. Zero Hidden Chain-of-Thought
To guarantee regulatory auditability and compliance:
- No free-form scratchpads, private thought blocks, or hidden chain-of-thought tokens are stored or transmitted.
- All reasoning is structured directly into explicit, standardized rubric dimensions.

---

## 3. Four Verdict Labels

The reasoner outputs one of four discrete verdicts from the `VerdictLabel` enum:

| Verdict | Definition | Criteria |
| :--- | :--- | :--- |
| `likely_reliable` | The statement appears credible and substantiated. | Clear attribution, internally consistent, supplied evidence directly supports key claims, no severe deception cues. |
| `misleading` | The statement contains distorted, exaggerated, or cherry-picked claims. | Discrepancies between claims and evidence, hyperbole, manipulated context, or partial support. |
| `likely_fake` | The statement presents fabricated or demonstrably contradicted information. | Direct contradiction by usable provenance-bearing evidence, complete fabrication, or severe deception indicators. |
| `unverifiable` | The truth value cannot be determined. | Essential evidence is absent, inconclusive, or conflicting; externally checkable claims lack verification. |

### The Unverifiable Policy
The reasoner **must** abstain with `unverifiable` whenever:
1. Externally checkable claims have zero supplied evidence (`evidence_support = "unavailable"`).
2. Available evidence is ambiguous, partial, or conflicting.
3. The statement consists of unverifiable predictions or subjective opinions lacking empirical backing.

---

## 4. Orthogonal Rubric Dimensions

The evaluation rubric assesses four independent dimensions to prevent conflating style, evidence, and structure:

### 1. Source Attribution (`source_attribution`)
Measures the transparency and specificity of cited sources *within the input text* (traceability, not verified truth):
- `strong`: A specific named source, identifiable institution, document, or directly attributed speaker is supplied.
- `partial`: Named individuals without official capacity, secondary reports, general organizational attribution.
- `weak`: Vague attribution (e.g., *"sources say"*, *"critics claim"*, *"insiders report"*).
- `none`: Completely anonymous assertion with no cited source or origin.
- `unknown`: Attribution status cannot be determined from the text snippet.

### 2. Evidence Support (`evidence_support`)
Measures the relationship between extracted claims and explicitly supplied evidence:
- `supported`: Supplied evidence directly substantiates all core claims.
- `contradicted`: Supplied evidence directly refutes or contradicts one or more core claims.
- `partial`: Evidence supports some claims but leaves others unsupported, or only partially corroborates them.
- `unavailable`: No external evidence was provided to check externally checkable claims.
- `not_applicable`: Statement consists purely of subjective opinions, self-evident definitions, or tautologies requiring no external evidence.

### 3. Internal Consistency (`internal_consistency`)
Measures logical coherence across claims strictly within the supplied material itself (never uses pretrained world knowledge to assess plausibility):
- `consistent`: No logical contradictions or conflicting statements within the text.
- `questionable`: Internal details contain contradictions, incompatible statements, or logical tensions strictly within the supplied material.
- `contradictory`: Direct mutual contradiction between statements within the same text.
- `unknown`: Inconclusive or too short to determine consistency.

### 4. Rhetorical Cues (`rhetorical_cues`)
A list of specific linguistic and presentation strategies detected in the text, such as:
- Clickbait phrasing, sensationalism, hyperbole
- Urgency triggers (*"SHARE BEFORE IT'S DELETED"*)
- Emotional appeals (fear, outrage, conspiracy)
- Loaded language or false dichotomies
- If the text is balanced, objective, and neutral, this field is an empty list `[]`.

---

## 5. Additional Output Fields

- **`unsupported_claims` (list of strings):** Specific claims from the input that are not substantiated by the supplied evidence. When no evidence is provided, all checkable claims are listed here.
- **`short_reason` (string, max 300 characters):** A concise, audit-ready justification summarizing the evidentiary basis of the assessment.
- **`prompt_version` (string):** The identifier of the reasoning prompt rubric (currently `reasoning_rubric_v1`).

---

## 6. Prompt Versioning & Storage

The system prompt is maintained in an external, versioned plain text file:
- **Location:** `backend/app/ai/prompts/reasoning_rubric_v1.txt`
- **Versioning Strategy:** Changes to the rubric or prompt structure require creating a new file (e.g., `reasoning_rubric_v2.txt`) and incrementing the version tag. Prompts are never hardcoded inside Python services.

---

## 7. Mock vs. Live Provider Behavior

| Dimension | `MockLLMProvider` | `GeminiProvider` (Live) |
| :--- | :--- | :--- |
| **Usage** | Automated CI/CD, offline unit tests (`pytest`), offline demos. | Production workflows, interactive API, live demos (`--live`). |
| **Model** | Internal deterministic pattern matcher. | `gemini-3.8-flash` via Google GenAI SDK. |
| **Network** | Zero network calls; 100% offline and deterministic. | HTTPS requests to Google Generative Language API. |
| **Evidence Adherence** | Programmatically maps archetypes (contradiction, support, empty evidence) to exact rubric values. | Instructed via `reasoning_rubric_v1.txt` with JSON schema enforcement. |
| **Safety Guarantees** | Immune to rate limits and API outages. | Bounded retries, exponential backoff, rate-limit awareness. |

---

## 8. Limitations & Boundary Rules

1. **Dependent on Upstream Inputs:** The accuracy of the reasoning depends entirely on the quality of extracted claims and retrieved evidence.
2. **No Autonomous Fact-Checking:** The reasoner cannot ascertain real-world truth if the retrieval step fails to provide relevant, high-quality reference snippets.
3. **Epistemic Humility:** When in doubt, the system must default to `unverifiable` rather than guessing or trusting model weights.
