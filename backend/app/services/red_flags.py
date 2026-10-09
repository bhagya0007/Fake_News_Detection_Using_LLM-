"""Interpretable Red-Flag and Stylometric Analysis Service (Step 8).

This module implements deterministic, rule-based detection of rhetorical,
stylistic, and presentation cues commonly scrutinized in misinformation analysis.

CORE DESIGN PRINCIPLES:
1. Warning signals only: Red flags are presentation cues, not proof of falsity.
2. No truth verdict: This service NEVER produces 'likely_fake', 'likely_reliable',
   'misleading', or 'unverifiable'. Verdicts are strictly prohibited at this layer.
3. Pure determinism: Uses lightweight regex and Python string methods. Zero ML models,
   zero LLMs, zero training, zero external network dependencies.
4. Acronym-safe capitalization: Does not flag short acronyms (e.g., 'NASA', 'WHO').
5. Explainable output: Every triggered flag includes a transparent explanation,
   severity level, and matched snippet.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configurable Thresholds
# ---------------------------------------------------------------------------
MIN_ALPHA_FOR_CAPS: int = 10
CAPS_RATIO_THRESHOLD: float = 0.50
HIGH_CAPS_RATIO_THRESHOLD: float = 0.80

EXCLAMATION_REPEAT_THRESHOLD: int = 2
HIGH_EXCLAMATION_TOTAL_THRESHOLD: int = 5

QUESTION_REPEAT_THRESHOLD: int = 2
HIGH_QUESTION_TOTAL_THRESHOLD: int = 4

EXCESSIVE_PUNCTUATION_RATIO_THRESHOLD: float = 0.15
MIN_PUNCTUATION_CHARS_FOR_RATIO: int = 4

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
SENSATIONAL_PHRASES: tuple[str, ...] = (
    "you won't believe",
    "you wont believe",
    "shocking",
    "explosive revelation",
    "secret they don't want you to know",
    "secret they dont want you to know",
    "exposed",
    "must see",
    "mind-blowing",
    "bombshell",
    "jaw-dropping",
    "breaking news",
    "breaking:",
    "miracle cure",
    "unbelievable truth",
    "mainstream media won't tell you",
    "mainstream media wont tell you",
    "what they are hiding",
)

CERTAINTY_PHRASES: tuple[str, ...] = (
    "100% proven",
    "100 percent proven",
    "definitely",
    "without a doubt",
    "guaranteed",
    "undeniable proof",
    "indisputable",
    "absolute truth",
    "proven beyond doubt",
    "irrefutable evidence",
    "completely proven",
)

URGENCY_PHRASES: tuple[str, ...] = (
    "share this before it is deleted",
    "share before it is deleted",
    "share this before they delete it",
    "share before they delete it",
    "share this before its deleted",
    "share before its deleted",
    "forward this",
    "spread this",
    "must share",
    "act now",
    "wake up people",
    "share immediately",
    "delete this",
    "pass this on",
    "share this",
)

ATTRIBUTION_PHRASES: tuple[str, ...] = (
    "according to",
    "reported by",
    "study by",
    "research by",
    "survey by",
    "official data",
    "government data",
    "census bureau",
    "department of",
    "study found",
    "researchers found",
    "scientists found",
    "analysts found",
    "announced",
    "stated",
    "spokesperson",
    "spokesman",
    "spokeswoman",
    "said",
    "says",
    "reported",
    "confirmed",
    "disclosed",
    "published in",
    "cited",
    "reuters",
    "associated press",
    "politifact",
)

# Regex Patterns
RE_URL = re.compile(r"https?://[^\s/$.?#].[^\s]*|www\.[^\s]+", re.IGNORECASE)
RE_PERCENTAGE = re.compile(
    r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:percent|percentage)\b", re.IGNORECASE
)
RE_NUMERIC_STAT = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|trillion|thousand|hundred|percent|%)?\b",
    re.IGNORECASE,
)
RE_REPEATED_EXCLAMATION = re.compile(r"!{2,}")
RE_REPEATED_QUESTION = re.compile(r"\?{2,}")
RE_MIXED_PUNCTUATION = re.compile(r"[!?]{2,}")
RE_QUOTES = re.compile(r'["“"”\'‘\'’]')
RE_DIRECT_ADDRESS = re.compile(r"\b(?:you|your|yours|you'll|you're)\b", re.IGNORECASE)


def extract_features(text: str) -> dict[str, Any]:
    """Extract quantitative stylometric and lexical feature counts from text.

    Args:
        text: Input statement string.

    Returns:
        Dictionary of raw measurable linguistic features.
    """
    char_count = len(text)
    words = text.split()
    word_count = len(words)

    # Uppercase analysis
    alpha_chars = [c for c in text if c.isalpha()]
    alpha_count = len(alpha_chars)
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    uppercase_ratio = round(upper_count / alpha_count, 4) if alpha_count > 0 else 0.0

    # Punctuation analysis
    exclamation_count = text.count("!")
    question_count = text.count("?")

    repeated_punct_matches = (
        len(RE_REPEATED_EXCLAMATION.findall(text))
        + len(RE_REPEATED_QUESTION.findall(text))
        + len(re.findall(r"\.{3,}", text))
    )

    punct_chars = [c for c in text if not c.isalnum() and not c.isspace()]
    punct_count = len(punct_chars)
    punct_ratio = round(punct_count / char_count, 4) if char_count > 0 else 0.0

    # Entity, statistical, and URL counts
    percentages = RE_PERCENTAGE.findall(text)
    percentage_count = len(percentages)

    stats = RE_NUMERIC_STAT.findall(text)
    # Filter stats to meaningful tokens (length > 0)
    statistical_expression_count = len([s for s in stats if s.strip()])

    urls = RE_URL.findall(text)
    url_count = len(urls)

    quotes = RE_QUOTES.findall(text)
    quote_count = len(quotes)

    # Phrase-level lexicon counts
    lower_text = text.lower()
    sensational_count = sum(1 for phrase in SENSATIONAL_PHRASES if phrase in lower_text)
    certainty_count = sum(1 for phrase in CERTAINTY_PHRASES if phrase in lower_text)
    urgency_count = sum(1 for phrase in URGENCY_PHRASES if phrase in lower_text)
    attribution_count = sum(1 for phrase in ATTRIBUTION_PHRASES if phrase in lower_text)

    return {
        "character_count": char_count,
        "word_count": word_count,
        "uppercase_letter_ratio": uppercase_ratio,
        "exclamation_count": exclamation_count,
        "question_mark_count": question_count,
        "repeated_punctuation_count": repeated_punct_matches,
        "percentage_count": percentage_count,
        "statistical_expression_count": statistical_expression_count,
        "url_count": url_count,
        "sensational_phrase_count": sensational_count,
        "certainty_phrase_count": certainty_count,
        "urgency_phrase_count": urgency_count,
        "attribution_phrase_count": attribution_count,
        "quote_count": quote_count,
        "excessive_punctuation_ratio": punct_ratio,
    }


def analyze_red_flags(text: str) -> dict[str, Any]:
    """Perform deterministic, interpretable red-flag and stylometric analysis.

    Validates input using the project's strict data policy, extracts quantitative
    features, and evaluates discrete rule-based triggers.

    Args:
        text: Raw statement string to evaluate.

    Returns:
        Structured dictionary containing:
        - 'flags': List of triggered flag dictionaries.
        - 'features': Quantitative stylometric feature counts.
        - 'summary': High-level flag count statistics.

    Raises:
        ValueError: If text is None, empty, or whitespace-only.
        TypeError: If text is not an instance of str.
    """
    if text is None:
        raise ValueError("Input text cannot be None.")

    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}.")

    if not text.strip():
        raise ValueError("Input text cannot be empty or whitespace-only.")

    features = extract_features(text)
    flags: list[dict[str, Any]] = []
    lower_text = text.lower()

    # 1. Excessive Caps (with acronym protection)
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) >= MIN_ALPHA_FOR_CAPS:
        if features["uppercase_letter_ratio"] >= HIGH_CAPS_RATIO_THRESHOLD:
            # Extract first uppercase snippet
            snippet = text[:min(60, len(text))].strip()
            flags.append(
                {
                    "type": "excessive_caps",
                    "severity": "high",
                    "message": "High proportion of uppercase letters detected throughout statement.",
                    "snippet": snippet,
                    "value": features["uppercase_letter_ratio"],
                }
            )
        elif features["uppercase_letter_ratio"] >= CAPS_RATIO_THRESHOLD:
            snippet = text[:min(60, len(text))].strip()
            flags.append(
                {
                    "type": "excessive_caps",
                    "severity": "medium",
                    "message": "Substantial proportion of uppercase letters detected.",
                    "snippet": snippet,
                    "value": features["uppercase_letter_ratio"],
                }
            )

    # 2. Excessive / Repeated Exclamation Marks
    repeated_excl = RE_REPEATED_EXCLAMATION.findall(text)
    if repeated_excl:
        snippet = repeated_excl[0]
        severity = "high" if len(snippet) >= 3 or features["exclamation_count"] >= HIGH_EXCLAMATION_TOTAL_THRESHOLD else "medium"
        flags.append(
            {
                "type": "excessive_exclamation",
                "severity": severity,
                "message": "Repeated or clustered exclamation marks detected.",
                "snippet": snippet,
                "value": features["exclamation_count"],
            }
        )
    elif features["exclamation_count"] >= 3:
        flags.append(
            {
                "type": "excessive_exclamation",
                "severity": "medium",
                "message": "Unusually high total count of exclamation marks detected.",
                "snippet": "!",
                "value": features["exclamation_count"],
            }
        )

    # 3. Repeated Question Marks
    repeated_q = RE_REPEATED_QUESTION.findall(text)
    if repeated_q:
        snippet = repeated_q[0]
        severity = "high" if len(snippet) >= 3 else "medium"
        flags.append(
            {
                "type": "repeated_question_marks",
                "severity": severity,
                "message": "Repeated or clustered question marks detected.",
                "snippet": snippet,
                "value": len(snippet),
            }
        )

    # 4. Sensational Language
    matched_sensational = [p for p in SENSATIONAL_PHRASES if p in lower_text]
    if matched_sensational:
        # Extract snippet matching the first occurrence in the original text
        p = matched_sensational[0]
        idx = lower_text.find(p)
        snippet = text[idx : idx + len(p)]
        severity = "high" if len(matched_sensational) >= 2 else "medium"
        flags.append(
            {
                "type": "sensational_language",
                "severity": severity,
                "message": "Sensational or hyperbolic presentation phrase detected.",
                "snippet": snippet,
                "value": len(matched_sensational),
            }
        )

    # 5. Certainty Language
    matched_certainty = [p for p in CERTAINTY_PHRASES if p in lower_text]
    if matched_certainty:
        p = matched_certainty[0]
        idx = lower_text.find(p)
        snippet = text[idx : idx + len(p)]
        severity = "medium" if len(matched_certainty) >= 2 else "low"
        flags.append(
            {
                "type": "certainty_language",
                "severity": severity,
                "message": "Rhetorical absolute certainty language detected (indicates style, not truth).",
                "snippet": snippet,
                "value": len(matched_certainty),
            }
        )

    # 6. Urgency or Share-Command Language
    matched_urgency = [p for p in URGENCY_PHRASES if p in lower_text]
    if matched_urgency:
        p = matched_urgency[0]
        idx = lower_text.find(p)
        snippet = text[idx : idx + len(p)]
        severity = "high" if len(matched_urgency) >= 2 else "medium"
        flags.append(
            {
                "type": "urgency_or_share_language",
                "severity": severity,
                "message": "Direct dissemination or urgency command detected.",
                "snippet": snippet,
                "value": len(matched_urgency),
            }
        )

    # 7. Statistical Claim (Informational / Low Severity)
    percentages = RE_PERCENTAGE.findall(text)
    if percentages:
        flags.append(
            {
                "type": "statistical_claim",
                "severity": "low",
                "message": "Explicit percentage assertion detected (informational signal).",
                "snippet": percentages[0],
                "value": len(percentages),
            }
        )

    # 8. Possible Missing Attribution
    # Cautious heuristic: Emitted only when quantitative assertions exist without attribution/quotes/URLs
    has_stat_assertions = features["percentage_count"] > 0 or features["statistical_expression_count"] >= 2
    if (
        has_stat_assertions
        and features["attribution_phrase_count"] == 0
        and features["url_count"] == 0
        and features["quote_count"] == 0
    ):
        flags.append(
            {
                "type": "possible_missing_attribution",
                "severity": "low",
                "message": (
                    "Statement asserts specific numerical or statistical claims without "
                    "explicit attribution language (e.g., 'according to', 'study shows')."
                ),
                "snippet": percentages[0] if percentages else "numerical assertion",
                "value": 0,
            }
        )

    # 9. Excessive Punctuation
    mixed_punct = RE_MIXED_PUNCTUATION.findall(text)
    has_mixed_clusters = any(("!" in m and "?" in m) for m in mixed_punct)
    punct_chars = [c for c in text if not c.isalnum() and not c.isspace()]
    if has_mixed_clusters:
        flags.append(
            {
                "type": "excessive_punctuation",
                "severity": "medium",
                "message": "Interrobang or mixed exclamation/question mark cluster detected.",
                "snippet": [m for m in mixed_punct if "!" in m and "?" in m][0],
                "value": len(punct_chars),
            }
        )
    elif (
        features["excessive_punctuation_ratio"] >= EXCESSIVE_PUNCTUATION_RATIO_THRESHOLD
        and len(punct_chars) >= MIN_PUNCTUATION_CHARS_FOR_RATIO
    ):
        flags.append(
            {
                "type": "excessive_punctuation",
                "severity": "medium",
                "message": "Unusually high density of punctuation characters detected.",
                "snippet": "".join(punct_chars[:10]),
                "value": features["excessive_punctuation_ratio"],
            }
        )

    # 10. Possible Clickbait (Compound Pattern)
    # Triggered when multiple rhetorical signals converge:
    # (Sensational phrase OR Excessive caps) + (Urgency OR Repeated punctuation OR Direct reader address)
    has_sensational_or_caps = bool(matched_sensational) or (
        len(alpha_chars) >= MIN_ALPHA_FOR_CAPS
        and features["uppercase_letter_ratio"] >= CAPS_RATIO_THRESHOLD
    )
    has_urgency_or_punct = bool(matched_urgency) or bool(repeated_excl) or bool(repeated_q)
    has_direct_address = bool(RE_DIRECT_ADDRESS.search(text))

    if has_sensational_or_caps and (has_urgency_or_punct or has_direct_address):
        # Determine clickbait severity based on signal density
        synergy_count = sum(
            [
                bool(matched_sensational),
                bool(matched_urgency),
                bool(repeated_excl or repeated_q),
                bool(has_direct_address),
                features["uppercase_letter_ratio"] >= CAPS_RATIO_THRESHOLD,
            ]
        )
        severity = "high" if synergy_count >= 3 else "medium"
        flags.append(
            {
                "type": "possible_clickbait",
                "severity": severity,
                "message": (
                    "Synergistic combination of sensational framing, emotional punctuation, "
                    "or direct reader engagement typical of clickbait."
                ),
                "snippet": matched_sensational[0] if matched_sensational else text[:30],
                "value": synergy_count,
            }
        )

    # Compile Summary
    high_count = sum(1 for f in flags if f["severity"] == "high")
    med_count = sum(1 for f in flags if f["severity"] == "medium")
    low_count = sum(1 for f in flags if f["severity"] == "low")

    summary = {
        "total_flags": len(flags),
        "high_severity_flags": high_count,
        "medium_severity_flags": med_count,
        "low_severity_flags": low_count,
        "has_attribution": features["attribution_phrase_count"] > 0,
        "has_statistical_claim": features["percentage_count"] > 0,
    }

    return {
        "flags": flags,
        "features": features,
        "summary": summary,
    }
