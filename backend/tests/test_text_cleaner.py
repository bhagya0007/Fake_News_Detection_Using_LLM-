"""Unit tests for text cleaning and normalization service.

Verifies:
1. Normal sentence remains unchanged.
2. Leading/trailing whitespace stripped.
3. Multiple internal spaces collapsed.
4. Newlines normalized to spaces.
5. Tabs normalized to spaces.
6. Capitalization preserved.
7. Exclamation marks preserved.
8. Question marks preserved.
9. Percentage values preserved.
10. Numbers preserved.
11. URLs preserved.
12. None input raises ValueError.
13. Non-string input raises TypeError.
14. Empty string raises ValueError.
15. Whitespace-only string raises ValueError.
16. DataFrame cleaning preserves original column and creates clean column without mutation.
"""

import pandas as pd
import pytest

from backend.app.services.text_cleaner import clean_dataframe_text, clean_text


def test_normal_sentence() -> None:
    """Test 1: Normal sentence without noise is returned unchanged."""
    raw = "This is a normal sentence."
    assert clean_text(raw) == "This is a normal sentence."


def test_leading_trailing_whitespace() -> None:
    """Test 2: Leading and trailing whitespace is stripped."""
    raw = "   This is a claim.   "
    assert clean_text(raw) == "This is a claim."


def test_multiple_spaces() -> None:
    """Test 3: Multiple consecutive spaces are collapsed to a single space."""
    raw = "This   is    a claim."
    assert clean_text(raw) == "This is a claim."


def test_newlines() -> None:
    """Test 4: Single and multiple newlines normalize to a single space."""
    raw = "This is\n\nan example."
    assert clean_text(raw) == "This is an example."


def test_tabs() -> None:
    """Test 5: Tabs normalize to single spaces safely."""
    raw = "This\tis\t\ta\tclaim."
    assert clean_text(raw) == "This is a claim."


def test_capitalization_preserved() -> None:
    """Test 6: All-caps words and casing are strictly preserved."""
    raw = "NASA CONFIRMED THIS"
    assert clean_text(raw) == "NASA CONFIRMED THIS"


def test_exclamation_marks_preserved() -> None:
    """Test 7: Repeated exclamation marks are preserved for sensationalism signals."""
    raw = "SHOCKING!!!"
    assert clean_text(raw) == "SHOCKING!!!"


def test_question_marks_preserved() -> None:
    """Test 8: Repeated question marks are preserved."""
    raw = "Really???"
    assert clean_text(raw) == "Really???"


def test_percentage_preserved() -> None:
    """Test 9: Numerical percentages with decimals are preserved."""
    raw = "Rates increased by 27.5%."
    assert clean_text(raw) == "Rates increased by 27.5%."


def test_numbers_preserved() -> None:
    """Test 10: Multi-digit numbers and years are preserved."""
    raw = "The event happened in 2024."
    assert clean_text(raw) == "The event happened in 2024."


def test_url_preserved() -> None:
    """Test 11: HTTP and HTTPS URLs are preserved intact."""
    raw = "Check https://example.com/facts?id=42 for info."
    assert clean_text(raw) == "Check https://example.com/facts?id=42 for info."


def test_none_input() -> None:
    """Test 12: None input raises ValueError."""
    with pytest.raises(ValueError, match="Input text cannot be None"):
        clean_text(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_val", [123, 45.6, ["some", "text"], {"key": "val"}])
def test_non_string_input(invalid_val: object) -> None:
    """Test 13: Non-string input types raise TypeError."""
    with pytest.raises(TypeError, match="Input text must be a string"):
        clean_text(invalid_val)  # type: ignore[arg-type]


def test_empty_string() -> None:
    """Test 14: Empty string raises ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        clean_text("")


@pytest.mark.parametrize("whitespace_val", ["   ", "\t\t", "\n\n", " \t \n "])
def test_whitespace_only_string(whitespace_val: str) -> None:
    """Test 15: Whitespace-only strings raise ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        clean_text(whitespace_val)


def test_clean_dataframe_text() -> None:
    """Test 16: DataFrame cleaning creates target column, preserves source, and does not mutate in place."""
    df = pd.DataFrame(
        {
            "id": ["1.json", "2.json"],
            "statement": ["  Claim   one!  ", "Claim\n\ntwo."],
        }
    )

    cleaned_df = clean_dataframe_text(df, source_column="statement", target_column="clean_statement")

    # Original DataFrame must not be mutated
    assert "clean_statement" not in df.columns
    assert df.loc[0, "statement"] == "  Claim   one!  "

    # New DataFrame has both columns intact and properly cleaned
    assert "clean_statement" in cleaned_df.columns
    assert cleaned_df.loc[0, "statement"] == "  Claim   one!  "
    assert cleaned_df.loc[0, "clean_statement"] == "Claim one!"
    assert cleaned_df.loc[1, "clean_statement"] == "Claim two."
