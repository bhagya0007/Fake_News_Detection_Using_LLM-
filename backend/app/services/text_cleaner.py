"""Safe Text Preprocessing Service.

This module provides conservative, non-destructive text cleaning for news claims.
It normalizes obvious whitespace, tabs, and newline formatting noise while strictly
preserving critical linguistic and stylistic signals:
- Punctuation (e.g. '!!!', '???') is retained for sensationalism detection.
- Capitalization (e.g. 'BREAKING NEWS') is retained for red-flag analysis.
- Numbers and percentages (e.g. '27.5%', '2024') are retained for factual claims.
- URLs and entity mentions are preserved.
- Stopwords, stemming, lemmatization, and lowercasing are deliberately NOT performed.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def clean_text(text: str) -> str:
    """Clean and normalize a single text string safely and deterministically.

    Performs conservative normalization:
    1. Validates string type and non-emptiness.
    2. Strips leading and trailing whitespace.
    3. Normalizes internal whitespace, newlines, and tabs into single ASCII spaces.
    4. Strips zero-width characters (e.g., zero-width space, BOM).
    5. Preserves all punctuation, numbers, capitalization, URLs, and vocabulary.

    Args:
        text: Input string to clean.

    Returns:
        Cleaned text string.

    Raises:
        ValueError: If text is None, empty, or whitespace-only.
        TypeError: If text is not an instance of str.
    """
    if text is None:
        raise ValueError("Input text cannot be None.")

    if not isinstance(text, str):
        raise TypeError(
            f"Input text must be a string, got {type(text).__name__}."
        )

    if not text.strip():
        raise ValueError("Input text cannot be empty or whitespace-only.")

    # Strip zero-width and invisible control characters
    sanitized = text.replace("\u200b", "").replace("\ufeff", "")

    # Normalize all whitespace (newlines, tabs, unicode non-breaking spaces) to single space
    cleaned = " ".join(sanitized.split())

    if not cleaned:
        raise ValueError("Input text cannot be empty or whitespace-only after cleaning.")

    return cleaned


def clean_dataframe_text(
    df: pd.DataFrame,
    source_column: str = "statement",
    target_column: str = "clean_statement",
) -> pd.DataFrame:
    """Apply clean_text to a DataFrame column without mutating the input DataFrame.

    Preserves the original source column alongside the newly created target column.

    Args:
        df: Input pandas DataFrame.
        source_column: Name of column containing raw text. Defaults to 'statement'.
        target_column: Name of new column for cleaned text. Defaults to 'clean_statement'.

    Returns:
        A new DataFrame containing all original columns plus target_column.

    Raises:
        ValueError: If df is not a DataFrame, is empty, or source_column is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty (0 rows).")

    if source_column not in df.columns:
        raise ValueError(
            f"Source column '{source_column}' not found in DataFrame columns: {list(df.columns)}."
        )

    df_out = df.copy()
    df_out[target_column] = df_out[source_column].apply(clean_text)

    # Sanity checks on target column
    if df_out[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains NaN values.")

    empty_count = int(
        (df_out[target_column].fillna("").astype(str).str.strip() == "").sum()
    )
    if empty_count > 0:
        raise ValueError(
            f"Target column '{target_column}' contains {empty_count} empty values."
        )

    return df_out
