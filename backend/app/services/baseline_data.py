"""Baseline Dataset Preparation Service.

Prepares reproducible, baseline-ready datasets for classical ML modeling:
- Loads canonical splits (train, valid, test) using dataset_loader.
- Applies label mapping (6 classes -> 3 project classes).
- Applies conservative text cleaning to create 'clean_statement' while preserving 'statement'.
- Enforces strict data-loss prevention and split integrity checks.
- Provides feature (X) and label (y) extraction utilities.

No vectorizers or ML models are instantiated or trained in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from backend.app.services.dataset_loader import (
    ALLOWED_PROJECT_LABELS,
    prepare_split,
)
from backend.app.services.text_cleaner import clean_dataframe_text

# Columns retained in the clean baseline view
BASELINE_DATASET_VIEW_COLUMNS: list[str] = [
    "id",
    "statement",
    "clean_statement",
    "label",
    "project_label",
]


def prepare_baseline_split(
    split_name: str, raw_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load, validate, map labels, and create clean text for a dataset split.

    Preserves the official split and all raw metadata.
    Enforces that zero rows are dropped.

    Args:
        split_name: Split identifier ('train', 'valid', 'validation', 'test').
        raw_dir: Optional custom path to data/raw directory.

    Returns:
        Prepared pandas DataFrame containing all original columns plus
        'project_label' and 'clean_statement'.

    Raises:
        ValueError: If split_name is invalid, rows are lost, or validation fails.
        FileNotFoundError: If the raw TSV file is missing.
    """
    # 1. Load, validate raw TSV, and apply label mapping
    df = prepare_split(split_name, raw_dir=raw_dir)
    initial_row_count = len(df)

    # 2. Apply non-destructive text cleaning
    prepared_df = clean_dataframe_text(
        df, source_column="statement", target_column="clean_statement"
    )

    # 3. Data Integrity & Loss Prevention Checks
    final_row_count = len(prepared_df)
    if final_row_count != initial_row_count:
        raise ValueError(
            f"Data loss detected in split '{split_name}': "
            f"started with {initial_row_count} rows, ended with {final_row_count} rows."
        )

    # 4. Feature and Label Completeness Checks
    if prepared_df["clean_statement"].isna().any():
        raise ValueError(
            f"Split '{split_name}' contains NaN values in 'clean_statement'."
        )

    empty_cleaned = int(
        (prepared_df["clean_statement"].fillna("").astype(str).str.strip() == "").sum()
    )
    if empty_cleaned > 0:
        raise ValueError(
            f"Split '{split_name}' contains {empty_cleaned} empty 'clean_statement' values."
        )

    if prepared_df["project_label"].isna().any():
        raise ValueError(
            f"Split '{split_name}' contains NaN values in 'project_label'."
        )

    # 5. Label Domain Validation
    observed_labels = set(prepared_df["project_label"].unique())
    if not observed_labels.issubset(ALLOWED_PROJECT_LABELS):
        invalid = observed_labels - ALLOWED_PROJECT_LABELS
        raise ValueError(
            f"Split '{split_name}' contains invalid project label(s): {invalid}."
        )

    if "unverifiable" in observed_labels:
        raise ValueError(
            f"Split '{split_name}' illegally contains 'unverifiable' label. "
            f"'unverifiable' must not be derived from LIAR ground truth."
        )

    return prepared_df


def prepare_all_baseline_splits(
    raw_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Prepare baseline datasets for all three canonical splits (train, valid, test).

    Maintains complete isolation between splits.

    Args:
        raw_dir: Optional custom raw directory path.

    Returns:
        Dictionary mapping 'train', 'valid', and 'test' to prepared DataFrames.
    """
    return {
        "train": prepare_baseline_split("train", raw_dir=raw_dir),
        "valid": prepare_baseline_split("valid", raw_dir=raw_dir),
        "test": prepare_baseline_split("test", raw_dir=raw_dir),
    }


def split_features_and_labels(
    df: pd.DataFrame,
    feature_col: str = "clean_statement",
    label_col: str = "project_label",
) -> tuple[pd.Series, pd.Series]:
    """Extract feature series (X) and label series (y) from a prepared DataFrame.

    Args:
        df: Prepared DataFrame containing feature_col and label_col.
        feature_col: Column name to extract as feature X. Defaults to 'clean_statement'.
        label_col: Column name to extract as label y. Defaults to 'project_label'.

    Returns:
        Tuple of (X, y) as pandas Series.

    Raises:
        KeyError: If feature_col or label_col is not present in df.
        ValueError: If X and y lengths do not match or are empty.
    """
    if feature_col not in df.columns:
        raise KeyError(
            f"Feature column '{feature_col}' not found in DataFrame."
        )

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    X = df[feature_col].copy()
    y = df[label_col].copy()

    if len(X) != len(y):
        raise ValueError(
            f"Length mismatch between features ({len(X)}) and labels ({len(y)})."
        )

    if len(X) == 0:
        raise ValueError("Cannot extract features and labels from an empty DataFrame.")

    return X, y


def get_baseline_dataset_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean view containing only baseline-relevant columns.

    Returns: ['id', 'statement', 'clean_statement', 'label', 'project_label'].

    Args:
        df: Prepared baseline DataFrame.

    Returns:
        Copy containing baseline view columns.

    Raises:
        KeyError: If any of the required view columns are missing.
    """
    for col in BASELINE_DATASET_VIEW_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' missing from DataFrame for baseline view."
            )

    return df[BASELINE_DATASET_VIEW_COLUMNS].copy()
