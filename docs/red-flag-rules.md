# Interpretable Red-Flag / Stylometric Rules Specification (Step 8)

> **Core Philosophy:**  
> **"A triggered red flag indicates a linguistic or presentation cue, not proof that the text is false."**  
> Sensational language can appear in true news. Percentages can be completely legitimate. Missing attribution is not proof that a statement is fabricated. This module provides interpretable warning signals that will later serve as one input to the hybrid fusion system. It does NOT decide truthfulness or generate verdicts.

---

## 1. Architectural Role & Principles

1. **Deterministic & Lightweight:** All detection logic is powered by transparent regular expressions and string analysis in [`backend/app/services/red_flags.py`](file:///e:/p105-fake-news-detection-llm-bhagya-varshney/backend/app/services/red_flags.py). Zero machine learning, zero LLMs, zero training, and zero heavy NLP dependencies.
2. **Strictly Non-Verbal:** The component never emits `likely_fake`, `likely_reliable`, `misleading`, or `unverifiable`.
3. **Severity Definition:**
   - **`low`:** Informational indicator (e.g., presence of percentage, missing attribution heuristic, single certainty phrase).
   - **`medium`:** Noticeable rhetorical intensity or formatting irregularity (e.g., repeated punctuation, moderate uppercase ratio, sensational phrase).
   - **`high`:** Extreme linguistic intensity or multi-signal clickbait synergy (e.g., multiple exclamation marks, heavy all-caps, multiple sensational/urgency phrases).
   - *Severity denotes linguistic intensity, NEVER probability of falsehood.*

---

## 2. Configuration & Threshold Reference

All thresholds are defined as named constants in `backend/app/services/red_flags.py`:

| Constant | Value | Purpose / Rationale |
| :--- | :---: | :--- |
| `MIN_ALPHA_FOR_CAPS` | `10` | Minimum alphabetic characters required before evaluating uppercase ratio. Prevents short acronyms ("NASA", "WHO", "UN") from falsely triggering caps flags. |
| `CAPS_RATIO_THRESHOLD` | `0.50` | Uppercase letter ratio threshold for `medium` severity excessive caps. |
| `HIGH_CAPS_RATIO_THRESHOLD` | `0.80` | Uppercase letter ratio threshold for `high` severity excessive caps. |
| `EXCLAMATION_REPEAT_THRESHOLD` | `2` | Number of consecutive exclamation marks (`!!`) required to trigger repeated exclamation flag. |
| `HIGH_EXCLAMATION_TOTAL_THRESHOLD` | `5` | Total exclamation mark count triggering `high` severity. |
| `QUESTION_REPEAT_THRESHOLD` | `2` | Number of consecutive question marks (`??`) required to trigger repeated question mark flag. |
| `HIGH_QUESTION_TOTAL_THRESHOLD` | `4` | Total question mark count triggering `high` severity. |
| `EXCESSIVE_PUNCTUATION_RATIO_THRESHOLD` | `0.15` | Punctuation-to-character ratio threshold for non-alphanumeric density. |
| `MIN_PUNCTUATION_CHARS_FOR_RATIO` | `4` | Minimum punctuation characters required before evaluating punctuation ratio. |

---

## 3. Detailed Rule Inventory

### Rule 1: `excessive_caps`
- **Signal:** High proportion of uppercase lettering.
- **Trigger Logic:** `len(alpha_chars) >= 10` AND `uppercase_ratio >= 0.50`.
- **Severity:** `high` if `uppercase_ratio >= 0.80`; `medium` otherwise.
- **Acronym Safe:** Short acronyms ("NASA launched a satellite") have `alpha_chars < 10` or ratio < 0.50 and are never flagged.
- **Snippet:** First 60 characters of the uppercase statement.

### Rule 2: `excessive_exclamation`
- **Signal:** Emotional, emphatic punctuation clusters.
- **Trigger Logic:** Regex `!{2,}` OR total exclamation count $\ge 3$.
- **Severity:** `high` if cluster length $\ge 3$ (e.g., `!!!`) or total $\ge 5$; `medium` otherwise.
- **Snippet:** Matched exclamation cluster (e.g., `!!!`).

### Rule 3: `repeated_question_marks`
- **Signal:** Rhetorical agitation or incredulity cues.
- **Trigger Logic:** Regex `\?{2,}`.
- **Severity:** `high` if cluster length $\ge 3$ (e.g., `???`); `medium` if `??`.
- **Snippet:** Matched question mark cluster (e.g., `???`).

### Rule 4: `sensational_language`
- **Signal:** Hyperbolic, tabloid, or conspiratorial buzzwords.
- **Trigger Logic:** Case-insensitive phrase match against `SENSATIONAL_PHRASES`.
- **Lexicon:** *"you won't believe", "shocking", "explosive revelation", "secret they don't want you to know", "exposed", "must see", "mind-blowing", "bombshell", "jaw-dropping", "breaking news", "breaking:", "miracle cure", "unbelievable truth", "mainstream media won't tell you", "what they are hiding"*.
- **Severity:** `high` if 2+ phrases match; `medium` if 1 phrase matches.
- **Snippet:** Matched phrase in original statement casing.

### Rule 5: `certainty_language`
- **Signal:** Unqualified rhetorical absolutism (indicates argumentative stance rather than empirical proof).
- **Trigger Logic:** Case-insensitive phrase match against `CERTAINTY_PHRASES`.
- **Lexicon:** *"100% proven", "100 percent proven", "definitely", "without a doubt", "guaranteed", "undeniable proof", "indisputable", "absolute truth", "proven beyond doubt", "irrefutable evidence", "completely proven"*.
- **Severity:** `medium` if 2+ phrases match; `low` if 1 phrase matches.
- **Snippet:** Matched certainty phrase.

### Rule 6: `urgency_or_share_language`
- **Signal:** Explicit calls for immediate viral dissemination or alarm.
- **Trigger Logic:** Case-insensitive phrase match against `URGENCY_PHRASES`.
- **Lexicon:** *"share before it is deleted", "share this before it is deleted", "share before they delete it", "forward this", "spread this", "must share", "act now", "wake up people", "share immediately", "delete this", "pass this on", "share this"*.
- **Severity:** `high` if 2+ phrases match; `medium` if 1 phrase matches.
- **Snippet:** Matched urgency phrase.

### Rule 7: `statistical_claim`
- **Signal:** Specific percentage assertions requiring empirical verification.
- **Trigger Logic:** Regex matching percentage patterns (e.g., `7.5%`, `12 percent`).
- **Severity:** Strictly `low` (Informational flag; numbers are not suspicious by default).
- **Snippet:** Matched percentage string (e.g., `"7.5%"`).

### Rule 8: `possible_missing_attribution`
- **Signal:** Specific quantitative assertions presented without attribution framing.
- **Trigger Logic:** Statement contains percentage or multiple numerical expressions (`percentage_count > 0` or `statistical_expression_count >= 2`) AND contains zero attribution phrases (`attribution_phrase_count == 0`), zero quotes, and zero URLs.
- **Attribution Lexicon:** *"according to", "reported by", "study by", "research by", "survey by", "official data", "government data", "census bureau", "study found", "announced", "stated", "said", "says", "reported", "confirmed", "politifact", "reuters", "associated press"*.
- **Severity:** Strictly `low` (Heuristic prompt for evidence check).
- **Snippet:** The numerical assertion lacking attribution.

### Rule 9: `excessive_punctuation`
- **Signal:** High density of punctuation characters or mixed interrobang clusters.
- **Trigger Logic:** Mixed clusters like `\?!` / `!\?` OR `punctuation_ratio >= 0.15` with at least 4 punctuation marks.
- **Severity:** `medium`.
- **Snippet:** Matched interrobang or punctuation cluster.

### Rule 10: `possible_clickbait`
- **Signal:** Compound synergy of sensational framing, emotional punctuation, and reader engagement.
- **Trigger Logic:** `(Sensational phrase OR Excessive caps)` AND `(Urgency phrase OR Repeated punctuation OR Direct reader address ("you", "your"))`.
- **Severity:** `high` if 3+ synergy signals coincide; `medium` if 2 coincide.
- **Snippet:** Key triggering sensational phrase.

---

## 4. Input Validation & Exception Handling

Input validation is consistent with the project's conservative preprocessing standards:
- `None` input $\rightarrow$ raises `ValueError("Input text cannot be None.")`
- Non-string input (e.g., `int`, `list`) $\rightarrow$ raises `TypeError("Input text must be a string...")`
- Empty or whitespace-only input $\rightarrow$ raises `ValueError("Input text cannot be empty or whitespace-only.")`

---

## 5. Empirical Training Set Observations

Descriptive evaluation on all 10,269 training statements (`models/red_flag_train_summary.json`):
- **Statements with any flag:** **15.56%**
- **Statistical assertions:** **37.94%** (highlights the prevalence of economic/policy figures in political discourse)
- **Attribution phrases:** **26.77%**
- **Sensational phrasing:** **0.05%** (rare in political discourse relative to social media)
- **Average flags per statement:** **0.226**
- **Critical Finding:** True statements (`likely_reliable`) averaged **0.278 flags per statement**, whereas false statements (`likely_fake`) averaged **0.156 flags per statement**. This occurs because reliable statements frequently cite verifiable statistical percentages (triggering informational `statistical_claim` flags). This provides empirical evidence in the LIAR training split that the implemented stylometric cues should not be used as a standalone proxy for falsehood.
