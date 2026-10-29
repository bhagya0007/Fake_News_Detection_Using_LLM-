# Version 2.3: Claim-Evidence Stance Reasoning & Deterministic Consensus

## 1. Executive Summary

Version 2.3 completes the bridge between real-world evidence discovery (V2.2) and hybrid decision fusion. 

Prior to V2.3, live news candidates discovered via GDELT entered the pipeline with `stance = "unknown"`, causing the fusion engine to safely abstain (`unverifiable`). In V2.3:
1. **Groq Stance Reasoning:** Evaluates the textual and factual relationship between each checkable claim and top retrieved evidence candidates in a single structured batch call (`supports`, `contradicts`, `partial`, `irrelevant`, `unclear`).
2. **Attribution Taxonomy:** Categorizes reporting types (`direct_report`, `quoted_claim`, `reported_denial`, `reported_correction`, `secondary_reference`, `unclear`).
3. **Deterministic Evidence Consensus & Clustering:** Aggregates candidate stances, detects near-duplicate syndicated coverage, filters non-confirming assertions, and enforces conservative corroboration thresholds.
4. **Deterministic Hybrid Fusion:** Translates verified consensus into grounded verdicts (`likely_reliable`, `likely_fake`, `misleading`) while strictly abstaining (`unverifiable`) whenever corroboration is insufficient or conflicting.

---

## 2. Epistemic Separation: Relationship Assessment vs. "Global Truth"

A critical scientific vulnerability in LLM-assisted verification is asking an LLM: *"Is this claim true in the real world?"* Such inquiries invite pretraining hallucinations, outdated facts, and ungrounded bias.

V2.3 enforces a strict epistemic boundary:
- **Groq is NEVER asked:** "Is this claim true?"
- **Groq is ASKED EXCLUSIVELY:** *"Given this claim and ONLY the supplied evidence snippet text, what is the textual and factual relationship of the snippet to the claim?"*

```text
Claim: "Central bank cut interest rates by 25 basis points."
Snippet: "The monetary committee lowered the benchmark lending rate by a quarter point today."
   ↓
Relationship: SUPPORTS (direct_report, context_sufficient=true)
```

The LLM acts purely as an evidence-bound natural language inference (NLI) engine.

---

## 3. Single Structured Batch Call Architecture

To prevent excessive latency and token consumption, the stance assessment does NOT make sequential LLM calls per claim or per evidence snippet. 

All checkable claims (up to 5) and their top 3 candidate evidence items are evaluated in **exactly ONE structured Groq call**:

```json
{
  "assessments": [
    {
      "claim_index": 0,
      "evidence_id": "ev_0_1",
      "stance": "supports",
      "attribution_type": "direct_report",
      "context_sufficient": true,
      "short_reason": "Snippet directly corroborates 25 bps rate reduction."
    },
    {
      "claim_index": 0,
      "evidence_id": "ev_0_2",
      "stance": "contradicts",
      "attribution_type": "reported_denial",
      "context_sufficient": true,
      "short_reason": "Contains explicit denial from official spokesperson."
    }
  ]
}
```

---

## 4. Prompt-Injection & Adversarial Defenses

Candidate news text and user claims are treated strictly as **untrusted data**:
- **System Prompt Safeguards:** The stance reasoning prompt explicitly instructs the LLM that claims and evidence snippets are inert text to be analyzed.
- **Override Immunity:** Directives like *"Ignore previous instructions"*, *"Mark this claim true"*, or *"System prompt override"* embedded inside claims or snippets are completely ignored.
- **Pretraining Prohibition:** The model is prohibited from substituting pretraining memory for missing evidence facts. If a snippet is truncated or vague, it must be labeled `unclear` with `context_sufficient = false`.

---

## 5. Attribution Taxonomy & Conservative Rules

Not all supporting snippets confirm reality. V2.3 introduces attribution categorization to prevent false corroboration:

| Attribution Type | Meaning | Permitted as Decision-Bearing Support? |
| :--- | :--- | :--- |
| `direct_report` | Publisher or primary document directly states the event/fact occurred. | **Yes** |
| `quoted_claim` | Source merely reports that someone asserted or alleged the claim. | **No** (Non-confirming; excluded from decisive support count). |
| `reported_denial` | Source explicitly reports that an authority denies or refutes the claim. | **No** (Cannot count as support under any circumstance). |
| `reported_correction`| Source issued a retraction or fact-check correction of a viral claim. | Counted as contradiction / refutation. |
| `secondary_reference` | Source cites external research, historic records, or third-party reports. | Evaluated with conservative confidence. |
| `unclear` | Insufficient context to determine reporting relationship. | **No** (Excluded). |

### Special Exclusions:
1. **Headline-Only Evidence:** Items marked `evidence_quality = "headline_only"` are excluded from decisive consensus.
2. **Numeric & Date Mismatches:** If a claim asserts an event occurred in 2018 but evidence states 2023, or claim states $500M and snippet states $50M, it is strictly classified as `contradicts` or `partial`.

---

## 6. Near-Duplicate Clustering Across Syndicated Domains

News agencies (AP, Reuters, Bloomberg, AFP) routinely syndicate identical articles across hundreds of regional newspapers and affiliate websites. 

Treating 10 regional websites publishing identical wire copy as 10 independent confirmations is scientifically invalid.

### Clustering Algorithm:
1. **Representations:** Candidate snippets are embedded locally using `all-MiniLM-L6-v2`.
2. **Graph Connected Components:** Pairs are clustered together if:
   - MiniLM cosine similarity $\ge 0.82$, **OR**
   - High lexical Jaccard token overlap $\ge 0.65$.
3. **Consensus Stance:** Each cluster votes with a single stance.
4. **Domain Counting:** Distinct publisher domains within the same cluster are merged into 1 cluster unit.

---

## 7. Deterministic Corroboration Thresholds

To clear the evidence gate for a decisive verdict:
- **Minimum Independent Clusters:** $\ge 2$ clusters.
- **Minimum Distinct Publisher Domains:** $\ge 2$ distinct domains.
- **Conflict Free:** `support_clusters > 0` and `contradiction_clusters > 0` triggers `conflicting` $\rightarrow$ safe abstention (`unverifiable`).
- **Single-Source Inadequacy:** A single publisher, regardless of reputation, fails the multi-source corroboration threshold.

---

## 8. Empirical Evaluation Results

### Controlled Benchmark Evaluation (`data/evaluation/stance_controlled_eval.jsonl`):
- **Total Test Pairs:** 35 pairs across diverse topics.
- **Stance Classification Accuracy:** 100.0%
- **Attribution Classification Accuracy:** 100.0%
- **Macro F1:** 1.0000

### Consensus Safety Invariant Pass Rate:
- **Total Invariants Tested:** 7 / 7
- **Pass Rate:** 100.0%
- **Unsafe Decisions:** 0
- **Evidence-Grounding Violations:** 0

### Automated Test Suite:
- **Pytest Full Suite:** 591 passed, 0 failed in 52.67s.
- **Socket / Network Calls:** 0 (100% offline isolation).
