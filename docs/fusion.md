# Hybrid Decision Fusion & Abstention Engine

This document details the architecture, rule hierarchy, evidence-gating mechanism, and safety policies of the Deterministic Hybrid Decision Fusion & Abstention Engine (`backend/app/services/fusion.py`) implemented in Step 14.

---

## 1. Purpose & Design Philosophy

The Hybrid Fusion Engine integrates multi-signal observations across the automated verification pipeline to produce a single authoritative `FinalAssessment`:
- `likely_reliable`
- `misleading`
- `likely_fake`
- `unverifiable`

### Core Principles
1. **NOT a Majority Vote:** Fusion is not an unweighted ensemble or majority voting system. True factual verification cannot be established by voting among pattern classifiers.
2. **Evidence-Gated Verification:** For externally checkable claims, usable provenance-bearing evidence availability acts as a mandatory gate. Without usable reference evidence, the system abstains with `unverifiable`.
3. **Abstention-First (Epistemic Humility):** In the presence of missing, ambiguous, or contradictory evidence, abstaining (`unverifiable` with `abstained=True`) is the mathematically and ethically correct decision.
4. **Deterministic & Auditable:** Fusion contains zero LLM calls, zero model inference, zero web requests, and zero randomized logic. Given identical inputs, it produces bitwise identical results in sub-millisecond time.

```text
ML Signal ─────────────┐
Red Flags ─────────────┤
Evidence Retrieval ────┼→ Fusion / Abstention Engine → FinalAssessment
LLM Assessment ────────┘
```

---

## 2. Component Signal Roles & Limitations

| Signal | Source | Role in Fusion | Limitations |
| :--- | :--- | :--- | :--- |
| **Classical ML Baseline** | TF-IDF + Logistic Regression | Secondary stylistic / linguistic pattern signal | Evaluated on frozen LIAR test split (Macro F1 = 0.4345). It is a text-pattern classifier and cannot verify facts. It never overrides evidence. |
| **Stylometric Red Flags** | Deterministic Regex Rules | Explanatory presentation cue & cautionary context | Detects sensationalism, urgency, and formatting anomalies. It can slightly reduce decision support score, but can never flip an evidence-grounded verdict. |
| **Evidence-Bound LLM Reasoner** | Structured Assessment (`LLMAssessment`) | Primary semantic reasoning over claims against evidence | Bound strictly to supplied evidence. Never uses pre-trained memory as evidence. |
| **Local Evidence Retrieval** | Local Evidence KB (`EvidenceRetrievalResult`) | Evidentiary Grounding Gate | Searches local validated KB. Topical similarity is decoupled from stance. |

---

## 3. Strict Rule Hierarchy

The engine evaluates rule-first deterministic logic before calculating numerical decision scores:

### Rule 1: No Checkable Factual Claim
If `has_checkable_claims == False`:
- Verdict: `unverifiable`, `abstained = True`.
- Reason: The statement consists of personal opinions, subjective aesthetic views, or non-verifiable speculative claims.

### Rule 2: No Usable Real Evidence (The Evidence Gate)
For checkable claims, if:
- Retrieved evidence is empty or below threshold (`no_sufficient_evidence == True`).
- All retrieved evidence items are synthetic fixtures (in production mode).
- All retrieved evidence items possess only `related` or `unknown` stance.
Then:
- Verdict: `unverifiable`, `abstained = True`.
- **The classical model and red-flag signals CANNOT override this rule.**

### Rule 3: LLM Reasoner Unavailable
If usable evidence exists, but the LLM reasoning component is unavailable (`llm is None` or failed):
- Verdict: `unverifiable`, `abstained = True`.
- The system abstains rather than making an ungrounded determination without reasoning.

### Rule 4: LLM Reasoner Concluded Unverifiable
If `llm.label == VerdictLabel.unverifiable`:
- Verdict: `unverifiable`, `abstained = True`.
- Respects LLM evidentiary abstention.

### Rule 5: Supported Evidence + Reliable LLM
If usable non-synthetic evidence contains supporting evidence (`stance == "supports"` or `llm.evidence_support == "supported"`) AND `llm.label == "likely_reliable"`:
- Verdict: `likely_reliable`, `abstained = False`.

### Rule 6: Contradicted Evidence + Fake LLM
If usable non-synthetic evidence contains refuting evidence (`stance == "contradicts"` or `llm.evidence_support == "contradicted"`) AND `llm.label == "likely_fake"`:
- Verdict: `likely_fake`, `abstained = False`.

### Rule 7: Partial Evidence + Misleading LLM
If `llm.evidence_support == "partial"` AND `llm.label == "misleading"`:
- Verdict: `misleading`, `abstained = False`.
- If partial evidence exists but LLM output is contradictory: safely abstains with `unverifiable`.

### Rule 8: Signal Disagreement Policy
- If Evidence/LLM says `likely_reliable`, but Classical ML predicts `likely_fake`:
  - Strong evidence takes precedence over the text-pattern classifier.
  - The system retains `likely_reliable`, records `ml_disagreement` in `signals_used`, and applies a penalty to `support_score`.
- If Evidence/LLM says `likely_fake`, but Classical ML predicts `likely_reliable`:
  - Retains `likely_fake`, records divergence, and reduces `support_score`.

### Rule 9: Red Flags Boundary Rule
- Red flags can **never** independently produce `likely_fake` or `misleading`.
- A sensational headline (e.g. `BREAKING NEWS!!! YOU WON'T BELIEVE...`) that is factually verified by official evidence remains `likely_reliable`. Red flags merely reduce the decision score slightly and attach an explanatory caution note.

---

## 4. Synthetic Evidence Quarantine Policy

To prevent test fixtures from corrupting real-world fact-checking:
- **Default Production Mode (`allow_synthetic_evidence=False`):** Any evidence record with `is_synthetic=True` is rejected during fusion. If only synthetic evidence is available, the system abstains.
- **Explicit Demo / Test Mode (`allow_synthetic_evidence=True`):** Allowed only in explicit integration tests and interactive demonstrations. If synthetic evidence is used, the verdict is explicitly prefixed with `[DEMO / SYNTHETIC EVIDENCE]`.

---

## 5. Support Score Meaning & Formulation

`FinalAssessment.support_score` is a bounded float strictly within $[0.0, 1.0]$.

> [!IMPORTANT]
> `support_score` is an **internal decision-support strength** calculated from multi-signal agreement and evidence grounding. It is **NOT an objective, calibrated truth probability** and must never be presented as a percentage likelihood of real-world veracity (e.g., *"87% probability of fake news"*).

### Mathematical Rules:
1. **When Abstained (`verdict == unverifiable`):** `support_score = 0.0`.
2. **When Definite Verdict (`likely_reliable`, `likely_fake`, `misleading`):**
   $$\text{score} = \text{base} (0.70) + \text{ML modifier} - \text{red-flag penalty}$$
   - **ML Agreement Bonus:** $+0.20 \times \max(P_{\text{ml}})$ if classical model agrees with evidence verdict.
   - **ML Disagreement Penalty:** $-0.15$ if classical model disagrees.
   - **Red-Flag Penalty:** $-0.05$ for 1 high-severity flag; $-0.10$ for $\ge 2$ high-severity flags.
   - Clamped strictly: $\text{score} = \max(0.0, \min(1.0, \text{score}))$.

---

## 6. Current Knowledge Base State & Expected Behavior

Because the production knowledge base (`data/evidence/evidence_kb.jsonl`) currently contains **0 provenance-bearing records** (held clean to prevent ungrounded citations):
> **In the current development state, most external factual claims will correctly result in `unverifiable` in production mode until usable provenance-bearing evidence records are seeded.**

This behavior is an intentional safety feature, adhering to the principle: **Quality > Coverage**.
