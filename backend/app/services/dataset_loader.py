"""Reusable LIAR Dataset Loader & Label Mapping Module.

This module provides data loading, schema and integrity validation,
and label mapping for the LIAR fake news benchmark dataset.

The original 6-class LIAR ratings are mapped into 3 project labels:
- true, mostly-true   -> likely_reliable
- half-true, barely-true -> misleading
- false, pants-fire   -> likely_fake

Note: The 4th project verdict ('unverifiable') is NOT derived from LIAR ground truth;
it is exclusively produced by subsequent inference/abstention logic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# Expected 14 columns in the original ACL 2017 LIAR benchmark specification
COLUMNS: list[str] = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]

# Canonical 6 original LIAR labels
ALLOWED_ORIGINAL_LABELS: set[str] = {
    "true",
    "mostly-true",
    "half-true",
    "barely-true",
    "false",
    "pants-fire",
}

# 3 target project labels for classical ML baseline & classification
ALLOWED_PROJECT_LABELS: set[str] = {
    "likely_reliable",
    "misleading",
    "likely_fake",
}

# Mapping from original 6-class ratings to 3-class project labels
LABEL_MAP: dict[str, str] = {
    "true": "likely_reliable",
    "mostly-true": "likely_reliable",
    "half-true": "misleading",
    "barely-true": "misleading",
    "false": "likely_fake",
    "pants-fire": "likely_fake",
}

# Recognized split identifiers and their corresponding filenames
SPLIT_FILE_MAP: dict[str, str] = {
    "train": "train.tsv",
    "valid": "valid.tsv",
    "validation": "valid.tsv",
    "test": "test.tsv",
}

# Baseline feature view columns
BASELINE_VIEW_COLUMNS: list[str] = [
    "id",
    "statement",
    "label",
    "project_label",
]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Determined relative to the location of this source module:
    backend/app/services/dataset_loader.py -> project root is 3 levels up from backend.
    """
    return Path(__file__).resolve().parents[3]


def resolve_split_name(split_name: str) -> str:
    """Validate and normalize a split identifier.

    Args:
        split_name: Name of the split ('train', 'valid', 'validation', 'test').

    Returns:
        Canonical split name ('train', 'valid', or 'test').

    Raises:
        ValueError: If split_name is not recognized.
    """
    if not isinstance(split_name, str):
        raise ValueError(
            f"Invalid split name type '{type(split_name).__name__}'. Expected string."
        )

    normalized = split_name.strip().lower()
    if normalized not in SPLIT_FILE_MAP:
        allowed = sorted(list(SPLIT_FILE_MAP.keys()))
        raise ValueError(
            f"Unknown split name '{split_name}'. Allowed splits: {allowed}."
        )

    # Normalize 'validation' alias to canonical 'valid'
    return "valid" if normalized == "validation" else normalized


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate DataFrame integrity, structure, statements, and labels.

    Args:
        df: pandas DataFrame loaded from a LIAR TSV file.

    Raises:
        ValueError: If the DataFrame is empty, has incorrect column counts,
            contains missing/empty statements, or has invalid/unknown labels.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError("Dataset DataFrame is empty (0 rows).")

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"DataFrame has {df.shape[1]} columns; expected exactly {len(COLUMNS)}."
        )

    if "statement" not in df.columns:
        raise ValueError("Required column 'statement' is missing from DataFrame.")

    if "label" not in df.columns:
        raise ValueError("Required column 'label' is missing from DataFrame.")

    # 1. Statement Validation: Missing (NaN)
    missing_stmts = int(df["statement"].isna().sum())
    if missing_stmts > 0:
        raise ValueError(
            f"Dataset contains {missing_stmts} missing (NaN) statement(s)."
        )

    # 2. Statement Validation: Empty or whitespace-only
    empty_stmts = int(
        (df["statement"].fillna("").astype(str).str.strip() == "").sum()
    )
    if empty_stmts > 0:
        raise ValueError(
            f"Dataset contains {empty_stmts} empty or whitespace-only statement(s)."
        )

    # 3. Label Validation: Missing (NaN)
    missing_labels = int(df["label"].isna().sum())
    if missing_labels > 0:
        raise ValueError(
            f"Dataset contains {missing_labels} missing (NaN) label(s)."
        )

    # 4. Label Validation: Unknown labels
    observed_labels = set(df["label"].unique())
    unknown_labels = observed_labels - ALLOWED_ORIGINAL_LABELS
    if unknown_labels:
        raise ValueError(
            f"Unknown label(s) detected: {sorted(list(unknown_labels))}. "
            f"Expected subset of: {sorted(list(ALLOWED_ORIGINAL_LABELS))}."
        )


def load_liar_split(
    split_name: str, raw_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load and validate an individual LIAR TSV split.

    Args:
        split_name: Name of the split ('train', 'valid', 'validation', 'test').
        raw_dir: Optional custom path to data/raw directory. Defaults to
            project_root / 'data' / 'raw'.

    Returns:
        Validated pandas DataFrame with original 14 columns.

    Raises:
        ValueError: If split_name is invalid or validation fails.
        FileNotFoundError: If the target TSV file does not exist.
    """
    canonical_split = resolve_split_name(split_name)

    if raw_dir is None:
        target_dir = get_project_root() / "data" / "raw"
    else:
        target_dir = Path(raw_dir)

    target_file = target_dir / SPLIT_FILE_MAP[canonical_split]
    if not target_file.is_file():
        raise FileNotFoundError(
            f"LIAR dataset file not found: {target_file.as_posix()}"
        )

    df = pd.read_csv(
        target_file,
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

    validate_dataframe(df)
    return df


def load_all_splits(
    raw_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Load all three canonical splits (train, valid, test).

    Args:
        raw_dir: Optional custom directory containing raw TSV files.

    Returns:
        Dictionary mapping 'train', 'valid', and 'test' to validated DataFrames.
    """
    return {
        "train": load_liar_split("train", raw_dir=raw_dir),
        "valid": load_liar_split("valid", raw_dir=raw_dir),
        "test": load_liar_split("test", raw_dir=raw_dir),
    }


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map original 6-class LIAR ratings to 3-class project labels.

    Creates a new column 'project_label' while preserving the original 'label'.
    Does not mutate the input DataFrame.

    Args:
        df: DataFrame containing a 'label' column with original LIAR categories.

    Returns:
        New DataFrame with added 'project_label' column.

    Raises:
        KeyError: If 'label' column is not in df.
        ValueError: If any label cannot be mapped or is unknown.
        RuntimeError: If 'unverifiable' is illegally introduced.
    """
    if "label" not in df.columns:
        raise KeyError("Column 'label' is required for label mapping.")

    observed_labels = set(df["label"].unique())
    unknown_labels = observed_labels - set(LABEL_MAP.keys())
    if unknown_labels:
        raise ValueError(
            f"Cannot map unknown label(s): {sorted(list(unknown_labels))}. "
            f"Expected subset of: {sorted(list(LABEL_MAP.keys()))}."
        )

    df_out = df.copy()
    df_out["project_label"] = df_out["label"].map(LABEL_MAP)

    # Sanity guard: 'unverifiable' must never be generated from LIAR ground truth
    if (df_out["project_label"] == "unverifiable").any():
        raise RuntimeError(
            "Illegal label mapping: 'unverifiable' must not be mapped from LIAR ground truth."
        )

    # Verify all mapped labels belong to ALLOWED_PROJECT_LABELS
    mapped_labels = set(df_out["project_label"].unique())
    invalid_project_labels = mapped_labels - ALLOWED_PROJECT_LABELS
    if invalid_project_labels:
        raise ValueError(
            f"Unexpected mapped project label(s): {sorted(list(invalid_project_labels))}."
        )

    return df_out


def prepare_split(
    split_name: str, raw_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load, validate, and apply label mapping to a dataset split.

    Args:
        split_name: Name of the split ('train', 'valid', 'validation', 'test').
        raw_dir: Optional custom raw directory.

    Returns:
        Prepared DataFrame containing 14 original columns plus 'project_label'.
    """
    df = load_liar_split(split_name, raw_dir=raw_dir)
    return map_labels(df)


def get_baseline_view(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a clean view containing only baseline-relevant columns.

    Returns columns: ['id', 'statement', 'label', 'project_label'].
    Does not mutate or drop columns from the original DataFrame.

    Args:
        df: Prepared DataFrame containing 'project_label'.

    Returns:
        A copy containing only baseline-relevant columns.

    Raises:
        KeyError: If any of the baseline columns are missing.
    """
    for col in BASELINE_VIEW_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' missing for baseline view. "
                f"Ensure map_labels() or prepare_split() has been called."
            )

    return df[BASELINE_VIEW_COLUMNS].copy()


def get_dataset_summary(
    df: pd.DataFrame, split_name: str = ""
) -> dict[str, Any]:
    """Generate a statistical summary of a loaded or prepared dataset split.

    Args:
        df: Dataset DataFrame.
        split_name: Optional identifier for the split.

    Returns:
        Dictionary with row count, original label distribution, and mapped
        project label distribution (if available).
    """
    summary: dict[str, Any] = {
        "split": split_name,
        "row_count": len(df),
        "original_labels": (
            df["label"].value_counts().to_dict() if "label" in df.columns else {}
        ),
    }

    if "project_label" in df.columns:
        summary["project_labels"] = df["project_label"].value_counts().to_dict()

    return summary
