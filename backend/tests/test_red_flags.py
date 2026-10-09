"""Unit tests for Interpretable Red-Flag and Stylometric Analysis (Step 8).

Verifies all 20 required test scenarios:
1. Normal neutral sentence yields few/no high-severity flags.
2. Excessive caps triggers caps and sensational signals.
3. Acronym protection prevents short uppercase words (e.g. 'NASA') from triggering excessive_caps.
4. Repeated exclamation marks trigger excessive_exclamation.
5. Repeated question marks trigger repeated_question_marks.
6. Sensational phrase triggers sensational_language.
7. Certainty phrase triggers certainty_language.
8. Urgency phrase triggers urgency_or_share_language.
9. Percentage detection identifies percentages without assigning a truth verdict.
10. Attribution detection suppresses possible_missing_attribution.
11. Unattributed statistical assertions emit possible_missing_attribution.
12. URL detection detects web links.
13. Numeric and statistical expression detection.
14. Punctuation-heavy text triggers excessive_punctuation.
15. Empty string raises ValueError.
16. None input raises ValueError.
17. Non-string input raises TypeError.
18. Output is strictly deterministic.
19. Returned structure strictly does NOT contain truth/fake verdicts.
20. Every flag object conforms to the required schema.
"""

from __future__ import annotations

import pytest

from backend.app.services.red_flags import (
    analyze_red_flags,
    extract_features,
)


def test_normal_neutral_sentence() -> None:
    """Test 1: Normal neutral sentence yields few/no high-severity flags."""
    text = "The central bank released its monthly report."
    res = analyze_red_flags(text)

    assert res["summary"]["high_severity_flags"] == 0
    assert res["features"]["exclamation_count"] == 0
    assert res["features"]["question_mark_count"] == 0


def test_excessive_caps() -> None:
    """Test 2: All-caps sentence triggers excessive_caps."""
    text = "THIS IS ABSOLUTELY SHOCKING NEWS"
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "excessive_caps" in flag_types
    caps_flag = [f for f in res["flags"] if f["type"] == "excessive_caps"][0]
    assert caps_flag["severity"] in {"medium", "high"}


def test_acronym_protection() -> None:
    """Test 3: Short uppercase acronyms do not trigger excessive_caps."""
    texts = [
        "NASA launched a new satellite today.",
        "The WHO announced updated international health guidance.",
        "UN delegates met in Geneva yesterday.",
    ]
    for t in texts:
        res = analyze_red_flags(t)
        flag_types = [f["type"] for f in res["flags"]]
        assert "excessive_caps" not in flag_types, f"Acronym in '{t}' falsely triggered excessive_caps!"


def test_repeated_exclamation() -> None:
    """Test 4: Repeated exclamation marks trigger excessive_exclamation."""
    text = "SHOCKING!!!"
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "excessive_exclamation" in flag_types
    excl_flag = [f for f in res["flags"] if f["type"] == "excessive_exclamation"][0]
    assert excl_flag["snippet"] == "!!!"
    assert excl_flag["severity"] == "high"


def test_repeated_question_marks() -> None:
    """Test 5: Repeated question marks trigger repeated_question_marks."""
    text = "Is this really true???"
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "repeated_question_marks" in flag_types
    q_flag = [f for f in res["flags"] if f["type"] == "repeated_question_marks"][0]
    assert q_flag["snippet"] == "???"
    assert q_flag["severity"] == "high"


def test_sensational_phrase() -> None:
    """Test 6: Sensational phrases trigger sensational_language."""
    text = "You won't believe what happened next."
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "sensational_language" in flag_types
    sens_flag = [f for f in res["flags"] if f["type"] == "sensational_language"][0]
    assert "you won't believe" in sens_flag["snippet"].lower()


def test_certainty_phrase() -> None:
    """Test 7: Absolute certainty language triggers certainty_language."""
    text = "This is 100% proven."
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "certainty_language" in flag_types
    cert_flag = [f for f in res["flags"] if f["type"] == "certainty_language"][0]
    assert "100% proven" in cert_flag["snippet"].lower()


def test_urgency_phrase() -> None:
    """Test 8: Urgency or call-to-share commands trigger urgency_or_share_language."""
    text = "Share this before it is deleted."
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "urgency_or_share_language" in flag_types
    urg_flag = [f for f in res["flags"] if f["type"] == "urgency_or_share_language"][0]
    assert "share" in urg_flag["snippet"].lower()
    assert "deleted" in urg_flag["snippet"].lower()


def test_percentage_detection() -> None:
    """Test 9: Percentage assertions are detected as statistical claims with low severity."""
    text = "According to the report, unemployment fell by 7.5%."
    res = analyze_red_flags(text)

    assert res["features"]["percentage_count"] == 1
    flag_types = [f["type"] for f in res["flags"]]
    assert "statistical_claim" in flag_types

    stat_flag = [f for f in res["flags"] if f["type"] == "statistical_claim"][0]
    assert stat_flag["severity"] == "low"
    assert "7.5%" in stat_flag["snippet"]

    # Verify no truth verdict is produced
    assert "predicted_label" not in res
    assert "verdict" not in res


def test_attribution_present_suppresses_missing_attribution() -> None:
    """Test 10: Statements with explicit attribution do not trigger possible_missing_attribution."""
    text = "According to the official report, inflation fell by 2%."
    res = analyze_red_flags(text)

    assert res["summary"]["has_attribution"] is True
    assert res["features"]["attribution_phrase_count"] >= 1
    flag_types = [f["type"] for f in res["flags"]]
    assert "possible_missing_attribution" not in flag_types


def test_attribution_absent_triggers_possible_missing_attribution() -> None:
    """Test 11: Quantitative assertions lacking attribution trigger possible_missing_attribution."""
    text = "Inflation fell by 73% last month."
    res = analyze_red_flags(text)

    assert res["summary"]["has_attribution"] is False
    flag_types = [f["type"] for f in res["flags"]]
    assert "possible_missing_attribution" in flag_types
    flag = [f for f in res["flags"] if f["type"] == "possible_missing_attribution"][0]
    assert flag["severity"] == "low"


def test_url_detection() -> None:
    """Test 12: URLs are detected and counted."""
    text = "Read the documentation at https://example.com/report for details."
    res = analyze_red_flags(text)

    assert res["features"]["url_count"] == 1


def test_numeric_preservation_and_detection() -> None:
    """Test 13: Numeric/statistical expressions are detected."""
    text = "Revenue increased from 10 to 25 million."
    res = analyze_red_flags(text)

    assert res["features"]["statistical_expression_count"] >= 2


def test_punctuation_heavy_example() -> None:
    """Test 14: Clustered interrobangs trigger excessive_punctuation."""
    text = "WHAT???!!!"
    res = analyze_red_flags(text)

    flag_types = [f["type"] for f in res["flags"]]
    assert "excessive_punctuation" in flag_types
    punct_flag = [f for f in res["flags"] if f["type"] == "excessive_punctuation"][0]
    assert punct_flag["severity"] == "medium"


def test_empty_input_raises_value_error() -> None:
    """Test 15: Empty string raises ValueError."""
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyze_red_flags("")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyze_red_flags("   \t\n  ")


def test_none_input_raises_value_error() -> None:
    """Test 16: None input raises ValueError."""
    with pytest.raises(ValueError, match="Input text cannot be None"):
        analyze_red_flags(None)  # type: ignore


def test_non_string_input_raises_type_error() -> None:
    """Test 17: Non-string input raises TypeError."""
    for invalid in [123, 45.6, ["text"], {"text": "val"}]:
        with pytest.raises(TypeError, match="must be a string"):
            analyze_red_flags(invalid)  # type: ignore


def test_deterministic_output() -> None:
    """Test 18: Identical inputs produce strictly identical results."""
    text = "SHOCKING!!! You won't believe what happened: 85% of scientists agree!"
    res1 = analyze_red_flags(text)
    res2 = analyze_red_flags(text)

    assert res1 == res2


def test_no_truth_verdict() -> None:
    """Test 19: Output strictly does NOT contain truth/fake classification verdicts."""
    text = "Unbelievable shocking news exposed by secret whistleblower!"
    res = analyze_red_flags(text)

    prohibited_keys = {
        "verdict",
        "predicted_label",
        "label",
        "likely_fake",
        "likely_reliable",
        "misleading",
        "unverifiable",
    }
    assert prohibited_keys.isdisjoint(res.keys())
    assert prohibited_keys.isdisjoint(res["summary"].keys())


def test_structured_flag_schema() -> None:
    """Test 20: Every emitted flag object contains all required fields."""
    text = "BREAKING: You won't believe this 100% proven secret they don't want you to know!!!"
    res = analyze_red_flags(text)

    assert len(res["flags"]) > 0
    required_flag_keys = {"type", "severity", "message", "snippet", "value"}

    for flag in res["flags"]:
        assert required_flag_keys.issubset(flag.keys())
        assert flag["severity"] in {"low", "medium", "high"}
        assert isinstance(flag["type"], str)
        assert isinstance(flag["message"], str)
        assert flag["snippet"] is not None
