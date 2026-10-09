"""Unit tests for baseline data preparation service.

Verifies:
1. Train split preparation succeeds.
2. Valid split preparation succeeds.
3. Test split preparation succeeds.
4. Row counts remain unchanged (0 data loss).
5. 'clean_statement' column exists and is non-empty.
6. Original 'statement' column remains intact.
7. 'project_label' column exists and is non-empty.
8. No missing values in cleaned text.
9. Only allowed 3 project labels exist ('likely_reliable', 'misleading', 'likely_fake').
10. 'unverifiable' does NOT appear in mapped ground truth.
11. Feature series X and label series y match in length.
12. Train, validation, and test splits remain completely separate.
13. Baseline view helper returns exact 5 expected columns.
"""

from pathlib import Path
import pytest
import pandas as pd

from backend.app.services.baseline_data import (
    BASELINE_DATASET_VIEW_COLUMNS,
    get_baseline_dataset_view,
    prepare_all_baseline_splits,
    prepare_baseline_split,
    split_features_and_labels,
)
from backend.app.services.dataset_loader import (
    ALLOWED_PROJECT_LABELS,
    load_all_splits,
)


@pytest.fixture(scope="module")
def prepared_splits() -> dict[str, pd.DataFrame]:
    """Fixture providing all prepared baseline splits."""
    return prepare_all_baseline_splits()


@pytest.fixture(scope="module")
def raw_splits() -> dict[str, pd.DataFrame]:
    """Fixture providing raw splits for comparison."""
    return load_all_splits()


def test_prepare_train_split(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 1: Verify train split prepares cleanly with 10,269 rows."""
    train_df = prepared_splits["train"]
    assert isinstance(train_df, pd.DataFrame)
    assert len(train_df) == 10269


def test_prepare_valid_split(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 2: Verify validation split prepares cleanly with 1,284 rows."""
    valid_df = prepared_splits["valid"]
    assert isinstance(valid_df, pd.DataFrame)
    assert len(valid_df) == 1284


def test_prepare_test_split(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 3: Verify test split prepares cleanly with 1,283 rows."""
    test_df = prepared_splits["test"]
    assert isinstance(test_df, pd.DataFrame)
    assert len(test_df) == 1283


def test_row_counts_unchanged(
    raw_splits: dict[str, pd.DataFrame],
    prepared_splits: dict[str, pd.DataFrame],
) -> None:
    """Test 4: Verify zero rows are lost between raw TSV loading and preparation."""
    for split_name in ["train", "valid", "test"]:
        assert len(raw_splits[split_name]) == len(prepared_splits[split_name]), (
            f"Row count mismatch in {split_name}"
        )


def test_clean_statement_exists(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 5: Verify 'clean_statement' column exists across all splits."""
    for split_name, df in prepared_splits.items():
        assert "clean_statement" in df.columns, (
            f"'clean_statement' missing in {split_name}"
        )


def test_original_statement_preserved(
    raw_splits: dict[str, pd.DataFrame],
    prepared_splits: dict[str, pd.DataFrame],
) -> None:
    """Test 6: Verify original 'statement' column remains available and intact."""
    for split_name, df in prepared_splits.items():
        assert "statement" in df.columns, f"'statement' missing in {split_name}"
        # Statement values match raw values
        assert (df["statement"] == raw_splits[split_name]["statement"]).all()


def test_project_label_exists(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 7: Verify 'project_label' column exists across all splits."""
    for split_name, df in prepared_splits.items():
        assert "project_label" in df.columns, (
            f"'project_label' missing in {split_name}"
        )


def test_no_missing_cleaned_text(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 8: Verify no NaN or empty values exist in 'clean_statement'."""
    for split_name, df in prepared_splits.items():
        assert int(df["clean_statement"].isna().sum()) == 0
        empty_count = int(
            (df["clean_statement"].fillna("").astype(str).str.strip() == "").sum()
        )
        assert empty_count == 0


def test_only_allowed_project_labels(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 9: Verify only allowed project labels exist in all splits."""
    for split_name, df in prepared_splits.items():
        observed = set(df["project_label"].unique())
        assert observed.issubset(ALLOWED_PROJECT_LABELS)
        assert observed == ALLOWED_PROJECT_LABELS


def test_unverifiable_does_not_appear(
    prepared_splits: dict[str, pd.DataFrame],
) -> None:
    """Test 10: Verify 'unverifiable' does NOT appear in any mapped LIAR split."""
    for split_name, df in prepared_splits.items():
        assert "unverifiable" not in df["project_label"].values


def test_split_features_and_labels(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 11: Verify split_features_and_labels returns matching X and y lengths."""
    for split_name, df in prepared_splits.items():
        X, y = split_features_and_labels(df)
        assert len(X) == len(y)
        assert len(X) == len(df)
        assert isinstance(X, pd.Series)
        assert isinstance(y, pd.Series)


def test_splits_remain_separate(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 12: Verify train, valid, and test DataFrames are distinct objects with disjoint IDs."""
    train_df = prepared_splits["train"]
    valid_df = prepared_splits["valid"]
    test_df = prepared_splits["test"]

    # Separate DataFrame references
    assert train_df is not valid_df
    assert train_df is not test_df
    assert valid_df is not test_df

    # Disjoint ID sets
    train_ids = set(train_df["id"])
    valid_ids = set(valid_df["id"])
    test_ids = set(test_df["id"])

    assert len(train_ids & valid_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(valid_ids & test_ids) == 0


def test_baseline_dataset_view(prepared_splits: dict[str, pd.DataFrame]) -> None:
    """Test 13: Verify baseline view extracts exactly the 5 target columns."""
    for split_name, df in prepared_splits.items():
        view_df = get_baseline_dataset_view(df)
        assert list(view_df.columns) == BASELINE_DATASET_VIEW_COLUMNS
        assert len(view_df) == len(df)
        # Original df maintains all 16 columns
        assert df.shape[1] == 16
