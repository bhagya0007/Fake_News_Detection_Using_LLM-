"""Dataset loader verification script.

Uses backend.app.services.dataset_loader to load, validate, map, and summarize
all canonical LIAR dataset splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.dataset_loader import (  # noqa: E402
    ALLOWED_PROJECT_LABELS,
    get_baseline_view,
    get_dataset_summary,
    prepare_split,
)


def print_split_summary(header: str, split_name: str) -> None:
    """Load, validate, map, and display statistics for a single split."""
    df = prepare_split(split_name)
    summary = get_dataset_summary(df, split_name=split_name)

    # Verify baseline view helper works
    baseline_df = get_baseline_view(df)
    assert len(baseline_df) == len(df), "Baseline view row count mismatch"

    # Verify 'unverifiable' is not in project labels
    assert "unverifiable" not in df["project_label"].values, (
        f"'unverifiable' illegally present in {split_name} project labels"
    )

    # Verify all project labels belong to the allowed 3 classes
    observed_project_labels = set(df["project_label"].unique())
    assert observed_project_labels.issubset(ALLOWED_PROJECT_LABELS), (
        f"Unexpected project labels in {split_name}: {observed_project_labels}"
    )

    total_rows = summary["row_count"]
    print(f"\n{header}")
    print(f"Rows: {total_rows}")

    print("Original labels:")
    for lbl, count in summary["original_labels"].items():
        pct = (count / total_rows) * 100
        print(f"  {lbl:16s}: {count:5d} ({pct:5.2f}%)")

    print("Mapped labels:")
    for lbl, count in summary["project_labels"].items():
        pct = (count / total_rows) * 100
        print(f"  {lbl:16s}: {count:5d} ({pct:5.2f}%)")


def main() -> None:
    """Run full loader verification across all splits."""
    print("=" * 80)
    print("LIAR DATASET LOADER & LABEL MAPPING VERIFICATION")
    print("=" * 80)

    # Load and verify each split
    print_split_summary("TRAIN", "train")
    print_split_summary("VALIDATION", "valid")
    print_split_summary("TEST", "test")

    print("\n" + "=" * 80)
    print("Dataset loader verification: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
