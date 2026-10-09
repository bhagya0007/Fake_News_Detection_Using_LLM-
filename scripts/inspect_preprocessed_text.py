"""Preprocessed Text Inspection & Statistics Script.

Loads train/valid/test through the baseline preparation pipeline, computes
comparative length statistics for raw vs cleaned statements, checks for data loss,
and verifies retention of punctuation, capitalization, numbers, and URLs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path for direct invocation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from backend.app.services.baseline_data import prepare_all_baseline_splits
from backend.app.services.dataset_loader import load_all_splits
from backend.app.services.text_cleaner import clean_text


def compute_length_stats(series: pd.Series) -> dict[str, dict[str, float]]:
    """Compute character and word length statistics for a text series."""
    texts = series.fillna("").astype(str)
    chars = texts.str.len()
    words = texts.str.split().str.len()
    return {
        "chars": {
            "min": float(chars.min()),
            "median": float(chars.median()),
            "mean": float(round(chars.mean(), 2)),
            "max": float(chars.max()),
        },
        "words": {
            "min": float(words.min()),
            "median": float(words.median()),
            "mean": float(round(words.mean(), 2)),
            "max": float(words.max()),
        },
    }


def verify_signal_preservation() -> None:
    """Verify that stylistic signals are strictly preserved through the cleaner."""
    test_cases = [
        ("Punctuation (exclamation)", "SHOCKING!!! Is this real???", "SHOCKING!!! Is this real???"),
        ("Numbers & percentages", "Unemployment grew by 27.5% in 2024.", "Unemployment grew by 27.5% in 2024."),
        ("Capitalization", "NASA CONFIRMED THIS FINDING", "NASA CONFIRMED THIS FINDING"),
        ("URL retention", "Read full report at https://example.org/study?id=123", "Read full report at https://example.org/study?id=123"),
        ("Quotes & currency", "He claimed: 'We invested $4.5 billion.'", "He claimed: 'We invested $4.5 billion.'"),
    ]

    print("\n" + "-" * 80)
    print("STYLISTIC SIGNAL PRESERVATION VERIFICATION")
    print("-" * 80)
    for name, raw_input, expected in test_cases:
        result = clean_text(raw_input)
        assert result == expected, f"Failed on {name}: got '{result}', expected '{expected}'"
        print(f"  [OK] {name:<30}: '{raw_input}' -> '{result}'")


def main() -> None:
    """Run full preprocessing inspection."""
    print("=" * 80)
    print("PREPROCESSED TEXT INSPECTION & STATISTICS")
    print("=" * 80)

    # 1. Load raw splits to verify data loss
    raw_splits = load_all_splits()
    prepared_splits = prepare_all_baseline_splits()

    print("\n[1] Checking Split Row Counts & Data Loss...")
    for split_name in ["train", "valid", "test"]:
        raw_count = len(raw_splits[split_name])
        prep_count = len(prepared_splits[split_name])
        print(
            f"  {split_name.capitalize():<6}: Raw rows = {raw_count:,} | Prepared rows = {prep_count:,}"
        )
        if raw_count != prep_count:
            raise RuntimeError(
                f"FAIL: Data loss detected in {split_name}! {raw_count} != {prep_count}"
            )
        print(f"  [OK] {split_name.capitalize()} row count preserved with 0 loss.")

    # 2. Changed statements during cleaning
    print("\n[2] Statement Modification Analysis...")
    changed_counts: dict[str, int] = {}
    for split_name in ["train", "valid", "test"]:
        df = prepared_splits[split_name]
        diff_mask = df["statement"] != df["clean_statement"]
        changed = int(diff_mask.sum())
        changed_counts[split_name] = changed
        pct = (changed / len(df)) * 100
        print(
            f"  {split_name.capitalize():<6}: {changed} of {len(df):,} statements changed ({pct:.2f}%)"
        )

    # 3. Statement Length Statistics Comparison
    print("\n[3] Text Length Statistics (Raw vs Cleaned)...")
    for split_name in ["train", "valid", "test"]:
        df = prepared_splits[split_name]
        raw_stats = compute_length_stats(df["statement"])
        clean_stats = compute_length_stats(df["clean_statement"])

        print(f"\n  --- {split_name.upper()} SPLIT ---")
        print(
            f"    Raw statement   : Chars [min={raw_stats['chars']['min']:.0f}, med={raw_stats['chars']['median']:.1f}, "
            f"mean={raw_stats['chars']['mean']:.2f}, max={raw_stats['chars']['max']:.0f}] | "
            f"Words [min={raw_stats['words']['min']:.0f}, med={raw_stats['words']['median']:.1f}, "
            f"mean={raw_stats['words']['mean']:.2f}, max={raw_stats['words']['max']:.0f}]"
        )
        print(
            f"    Clean statement : Chars [min={clean_stats['chars']['min']:.0f}, med={clean_stats['chars']['median']:.1f}, "
            f"mean={clean_stats['chars']['mean']:.2f}, max={clean_stats['chars']['max']:.0f}] | "
            f"Words [min={clean_stats['words']['min']:.0f}, med={clean_stats['words']['median']:.1f}, "
            f"mean={clean_stats['words']['mean']:.2f}, max={clean_stats['words']['max']:.0f}]"
        )

    # 4. Sample Before/After Records
    print("\n[4] Sample Before / After Records:")
    for split_name in ["train", "valid", "test"]:
        df = prepared_splits[split_name]
        print(f"\n  --- Sample from {split_name.upper()} ---")
        for _, row in df.head(2).iterrows():
            stmt = row["statement"][:70] + "..." if len(row["statement"]) > 70 else row["statement"]
            c_stmt = row["clean_statement"][:70] + "..." if len(row["clean_statement"]) > 70 else row["clean_statement"]
            print(f"    [{row['id']}] ({row['project_label']})")
            print(f"      Raw   : \"{stmt}\"")
            print(f"      Clean : \"{c_stmt}\"")

    # 5. Signal Preservation Tests
    verify_signal_preservation()

    print("\n" + "=" * 80)
    print("Preprocessed text inspection: PASS (0 rows lost, stylistic signals intact)")
    print("=" * 80)


if __name__ == "__main__":
    main()
