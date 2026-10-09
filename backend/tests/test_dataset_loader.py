"""Comprehensive unit tests for LIAR dataset loader and label mapping.

Test Coverage:
1. Load train split
2. Load validation split (including 'validation' alias)
3. Load test split
4. Expected 14 columns present in loaded splits
5. Label mapping: true -> likely_reliable
6. Label mapping: mostly-true -> likely_reliable
7. Label mapping: half-true -> misleading
8. Label mapping: barely-true -> misleading
9. Label mapping: false -> likely_fake
10. Label mapping: pants-fire -> likely_fake
11. Unknown label error handling
12. Invalid split name error handling
13. Missing file handling (using tmp_path, without touching real data)
14. Empty and whitespace-only statement validation
15. Original 'label' column preservation
16. Project label set integrity on real data
17. Absolute exclusion of 'unverifiable' from mapped labels
18. Baseline view extraction and immutability
"""

from pathlib import Path
import pytest
import pandas as pd

from backend.app.services.dataset_loader import (
    ALLOWED_ORIGINAL_LABELS,
    ALLOWED_PROJECT_LABELS,
    BASELINE_VIEW_COLUMNS,
    COLUMNS,
    LABEL_MAP,
    get_baseline_view,
    get_dataset_summary,
    load_all_splits,
    load_liar_split,
    map_labels,
    prepare_split,
    resolve_split_name,
    validate_dataframe,
)


@pytest.fixture(scope="module")
def raw_data_dir() -> Path:
    """Return path to data/raw directory relative to repository root."""
    return Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.fixture(scope="module")
def sample_valid_row() -> dict:
    """Fixture providing a valid 14-column dummy row for testing edge cases."""
    return {
        "id": "test_001.json",
        "label": "true",
        "statement": "This is a legitimate test statement.",
        "subjects": "politics",
        "speaker": "John Doe",
        "speaker_job": "Senator",
        "state": "Virginia",
        "party": "independent",
        "barely_true_count": 0,
        "false_count": 0,
        "half_true_count": 0,
        "mostly_true_count": 1,
        "pants_fire_count": 0,
        "context": "press release",
    }


# ==============================================================================
# 1-3. Split Loading Tests
# ==============================================================================

def test_load_train(raw_data_dir: Path) -> None:
    """Test 1: Verify train split loads successfully with expected dimensions."""
    df = load_liar_split("train", raw_dir=raw_data_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10269
    assert df.shape[1] == 14


def test_load_validation(raw_data_dir: Path) -> None:
    """Test 2: Verify valid split loads successfully, including alias handling."""
    df_valid = load_liar_split("valid", raw_dir=raw_data_dir)
    assert isinstance(df_valid, pd.DataFrame)
    assert len(df_valid) == 1284
    assert df_valid.shape[1] == 14

    # Test 'validation' alias
    df_alias = load_liar_split("validation", raw_dir=raw_data_dir)
    assert len(df_alias) == len(df_valid)


def test_load_test(raw_data_dir: Path) -> None:
    """Test 3: Verify test split loads successfully with expected dimensions."""
    df = load_liar_split("test", raw_dir=raw_data_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1283
    assert df.shape[1] == 14


# ==============================================================================
# 4. Column Schema Tests
# ==============================================================================

def test_expected_columns(raw_data_dir: Path) -> None:
    """Test 4: Verify all 14 original columns are present across all splits."""
    splits = load_all_splits(raw_dir=raw_data_dir)
    for name, df in splits.items():
        assert list(df.columns) == COLUMNS, f"Column mismatch in split '{name}'"


# ==============================================================================
# 5-10. Individual Label Mapping Tests
# ==============================================================================

@pytest.mark.parametrize(
    "orig_label, expected_project_label",
    [
        ("true", "likely_reliable"),
        ("mostly-true", "likely_reliable"),
        ("half-true", "misleading"),
        ("barely-true", "misleading"),
        ("false", "likely_fake"),
        ("pants-fire", "likely_fake"),
    ],
)
def test_label_mapping_individual(
    sample_valid_row: dict, orig_label: str, expected_project_label: str
) -> None:
    """Tests 5-10: Verify mapping for each of the 6 original LIAR labels."""
    row = dict(sample_valid_row)
    row["label"] = orig_label
    df = pd.DataFrame([row])

    mapped_df = map_labels(df)
    assert mapped_df.iloc[0]["project_label"] == expected_project_label


# ==============================================================================
# 11-13. Error Handling Tests (Unknown Label, Bad Split, Missing File)
# ==============================================================================

def test_unknown_label_raises_error(sample_valid_row: dict) -> None:
    """Test 11: A fake/unknown label must raise a clear ValueError."""
    row = dict(sample_valid_row)
    row["label"] = "completely_fabricated_label"
    df = pd.DataFrame([row])

    with pytest.raises(ValueError, match="Cannot map unknown label"):
        map_labels(df)

    with pytest.raises(ValueError, match="Unknown label"):
        validate_dataframe(df)


def test_invalid_split_name_raises_error() -> None:
    """Test 12: Invalid split names such as 'foo' must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown split name 'foo'"):
        resolve_split_name("foo")

    with pytest.raises(ValueError, match="Unknown split name 'unknown_split'"):
        load_liar_split("unknown_split")


def test_missing_file_raises_error(tmp_path: Path) -> None:
    """Test 13: Missing file must raise FileNotFoundError (tested using isolated tmp_path)."""
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="LIAR dataset file not found"):
        load_liar_split("train", raw_dir=empty_dir)


# ==============================================================================
# 14. Empty and Whitespace Statement Validation
# ==============================================================================

@pytest.mark.parametrize(
    "bad_statement",
    [
        None,
        "",
        "   ",
        "\t\n  \t",
    ],
)
def test_empty_statement_validation(
    sample_valid_row: dict, bad_statement: str | None
) -> None:
    """Test 14: NaN, empty, or whitespace-only statements must raise ValueError."""
    row = dict(sample_valid_row)
    row["statement"] = bad_statement
    df = pd.DataFrame([row])

    with pytest.raises(ValueError, match="statement"):
        validate_dataframe(df)


# ==============================================================================
# 15. Original Label Preservation
# ==============================================================================

def test_original_label_preservation(sample_valid_row: dict) -> None:
    """Test 15: Verify that map_labels creates 'project_label' without modifying 'label'."""
    row = dict(sample_valid_row)
    row["label"] = "mostly-true"
    df = pd.DataFrame([row])

    mapped_df = map_labels(df)

    # Input DataFrame must not be mutated
    assert "project_label" not in df.columns

    # Output DataFrame has both columns intact
    assert mapped_df.iloc[0]["label"] == "mostly-true"
    assert mapped_df.iloc[0]["project_label"] == "likely_reliable"


# ==============================================================================
# 16-17. Project Label Set and Unverifiable Guard
# ==============================================================================

def test_project_label_set(raw_data_dir: Path) -> None:
    """Test 16: Verify that all project labels in real mapped training data belong to allowed set."""
    train_df = prepare_split("train", raw_dir=raw_data_dir)
    observed = set(train_df["project_label"].unique())

    assert observed.issubset(ALLOWED_PROJECT_LABELS)
    assert observed == {"likely_reliable", "misleading", "likely_fake"}


def test_no_unverifiable(raw_data_dir: Path) -> None:
    """Test 17: Verify 'unverifiable' does NOT appear in mapped LIAR labels in any split."""
    for split in ["train", "valid", "test"]:
        df = prepare_split(split, raw_dir=raw_data_dir)
        assert "unverifiable" not in df["project_label"].values


# ==============================================================================
# 18. Baseline View & Summary Helper Tests
# ==============================================================================

def test_baseline_view_and_summary(raw_data_dir: Path) -> None:
    """Test 18: Verify baseline view returns exact required columns without dropping original data."""
    train_df = prepare_split("train", raw_dir=raw_data_dir)
    view_df = get_baseline_view(train_df)

    assert list(view_df.columns) == BASELINE_VIEW_COLUMNS
    assert len(view_df) == len(train_df)
    # Original train_df retained all 15 columns
    assert train_df.shape[1] == 15

    # Test summary helper
    summary = get_dataset_summary(train_df, split_name="train")
    assert summary["row_count"] == 10269
    assert "likely_reliable" in summary["project_labels"]
    assert "misleading" in summary["project_labels"]
    assert "likely_fake" in summary["project_labels"]
