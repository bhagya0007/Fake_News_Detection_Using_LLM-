"""Unit tests for LIAR dataset inspection and integrity validation.

Verifies:
1. All expected dataset files exist.
2. Each file has exactly 14 columns.
3. Every label belongs to the allowed benchmark set.
4. No missing labels in any split.
5. No empty statements in the training set.
"""

from pathlib import Path
import pytest
import pandas as pd

from scripts.inspect_dataset import (
    ALLOWED_LABELS,
    COLUMNS,
    EXPECTED_FILES,
    check_files_exist,
    inspect_empty_statements,
    load_all_splits,
)


@pytest.fixture(scope="module")
def raw_data_dir() -> Path:
    """Fixture to obtain data/raw directory relative to repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "raw"


@pytest.fixture(scope="module")
def dataset_splits(raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    """Fixture to load all splits once for test module."""
    return load_all_splits(raw_data_dir)


def test_all_expected_dataset_files_exist(raw_data_dir: Path) -> None:
    """Test 1: Verify all expected TSV files exist on disk."""
    file_map = check_files_exist(raw_data_dir)
    for expected_file in EXPECTED_FILES:
        assert expected_file in file_map
        assert file_map[expected_file].is_file()
        assert file_map[expected_file].stat().st_size > 0


def test_each_file_has_14_columns(
    dataset_splits: dict[str, pd.DataFrame],
) -> None:
    """Test 2: Verify each split DataFrame has exactly 14 columns matching specifications."""
    for split_name, df in dataset_splits.items():
        assert (
            df.shape[1] == 14
        ), f"Split '{split_name}' has {df.shape[1]} columns, expected 14."
        assert list(df.columns) == COLUMNS


def test_every_label_belongs_to_allowed_set(
    dataset_splits: dict[str, pd.DataFrame],
) -> None:
    """Test 3: Verify that every observed label is one of the 6 benchmark categories."""
    for split_name, df in dataset_splits.items():
        observed_labels = set(df["label"].unique())
        unknown_labels = observed_labels - ALLOWED_LABELS
        assert (
            not unknown_labels
        ), f"Split '{split_name}' contains unknown labels: {unknown_labels}"


def test_no_missing_labels(dataset_splits: dict[str, pd.DataFrame]) -> None:
    """Test 4: Verify there are zero missing (NaN) labels across all splits."""
    for split_name, df in dataset_splits.items():
        missing_count = int(df["label"].isna().sum())
        assert (
            missing_count == 0
        ), f"Split '{split_name}' contains {missing_count} missing labels."


def test_no_empty_training_statements(
    dataset_splits: dict[str, pd.DataFrame],
) -> None:
    """Test 5: Verify there are zero NaN, empty, or whitespace-only statements in train."""
    train_df = dataset_splits["train"]
    empty_count = inspect_empty_statements(train_df)
    assert (
        empty_count == 0
    ), f"Train split contains {empty_count} empty or whitespace-only statements."
