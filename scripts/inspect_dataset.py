"""LIAR Dataset Inspection & Validation Script.

This script inspects, validates, and reports on the raw LIAR dataset TSV files
(train.tsv, valid.tsv, test.tsv) without modifying the raw data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# Expected 14 columns defined in the original ACL 2017 LIAR benchmark specification
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

ALLOWED_LABELS: set[str] = {
    "true",
    "mostly-true",
    "half-true",
    "barely-true",
    "false",
    "pants-fire",
}

EXPECTED_FILES: list[str] = ["train.tsv", "valid.tsv", "test.tsv"]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def check_files_exist(raw_dir: Path) -> dict[str, Path]:
    """Verify that all required TSV dataset files exist.

    Args:
        raw_dir: Path to data/raw directory.

    Returns:
        Dictionary mapping filename to Path.

    Raises:
        FileNotFoundError: If any expected dataset file is missing.
    """
    file_paths: dict[str, Path] = {}
    for filename in EXPECTED_FILES:
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required dataset file missing: {path.as_posix()}"
            )
        file_paths[filename] = path
    return file_paths


def load_tsv_split(file_path: Path) -> pd.DataFrame:
    """Load and validate an individual TSV split.

    Args:
        file_path: Path to the TSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        ValueError: If column count does not match the 14 expected columns.
    """
    # Use csv.QUOTE_NONE to avoid misinterpreting quotation marks inside statements
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"File '{file_path.name}' has {df.shape[1]} columns; expected {len(COLUMNS)}."
        )

    return df


def load_all_splits(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load train, valid, and test splits into DataFrames.

    Args:
        raw_dir: Path to directory containing raw TSV files.

    Returns:
        Dict mapping split names ('train', 'valid', 'test') to DataFrames.
    """
    file_paths = check_files_exist(raw_dir)
    splits: dict[str, pd.DataFrame] = {
        "train": load_tsv_split(file_paths["train.tsv"]),
        "valid": load_tsv_split(file_paths["valid.tsv"]),
        "test": load_tsv_split(file_paths["test.tsv"]),
    }
    return splits


def inspect_label_distribution(
    df: pd.DataFrame, split_name: str
) -> dict[str, Any]:
    """Analyze label counts, percentages, and check for unknown labels."""
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    observed_labels = set(counts.keys())
    unknown_labels = observed_labels - ALLOWED_LABELS

    percentages = {
        lbl: (count / total) * 100 for lbl, count in counts.items()
    }

    return {
        "split": split_name,
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "observed_labels": observed_labels,
        "unknown_labels": unknown_labels,
    }


def inspect_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Count missing (NaN) values per column."""
    return df.isna().sum().to_dict()


def inspect_empty_statements(df: pd.DataFrame) -> int:
    """Detect empty, NaN, or whitespace-only statement strings."""
    is_na = df["statement"].isna()
    is_empty_or_ws = df["statement"].fillna("").astype(str).str.strip() == ""
    return int((is_na | is_empty_or_ws).sum())


def inspect_within_split_duplicates(
    df: pd.DataFrame,
) -> dict[str, int]:
    """Count duplicate IDs and duplicate statements within a single split."""
    dup_ids = int(df["id"].duplicated().sum())
    dup_statements = int(df["statement"].duplicated().sum())
    return {
        "duplicate_ids": dup_ids,
        "duplicate_statements": dup_statements,
    }


def inspect_cross_split_overlap(
    splits: dict[str, pd.DataFrame],
) -> dict[str, dict[str, int]]:
    """Check ID and statement overlap across dataset splits."""
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    overlap_results: dict[str, dict[str, int]] = {}

    for split_a, split_b in pairs:
        set_id_a = set(splits[split_a]["id"])
        set_id_b = set(splits[split_b]["id"])
        set_stmt_a = set(splits[split_a]["statement"])
        set_stmt_b = set(splits[split_b]["statement"])

        id_overlap = len(set_id_a & set_id_b)
        stmt_overlap = len(set_stmt_a & set_stmt_b)

        overlap_results[f"{split_a}_vs_{split_b}"] = {
            "id_overlap": id_overlap,
            "statement_overlap": stmt_overlap,
        }

    return overlap_results


def inspect_text_lengths(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute character and word length statistics for the statement field."""
    statements = df["statement"].fillna("").astype(str)
    char_lens = statements.str.len()
    word_lens = statements.str.split().str.len()

    return {
        "characters": {
            "min": float(char_lens.min()),
            "median": float(char_lens.median()),
            "mean": float(round(char_lens.mean(), 2)),
            "max": float(char_lens.max()),
        },
        "words": {
            "min": float(word_lens.min()),
            "median": float(word_lens.median()),
            "mean": float(round(word_lens.mean(), 2)),
            "max": float(word_lens.max()),
        },
    }


def evaluate_data_quality(
    files_ok: bool,
    columns_ok: bool,
    unknown_labels_count: int,
    missing_critical_count: int,
    duplicate_ids_count: int,
    within_split_duplicate_statements: int,
    cross_split_statement_overlap: int,
) -> dict[str, str]:
    """Produce PASS, WARNING, or FAIL status for each dataset quality category."""
    evaluations: dict[str, str] = {}

    evaluations["Files present"] = "PASS" if files_ok else "FAIL"
    evaluations["Column structure"] = "PASS" if columns_ok else "FAIL"
    evaluations["Known labels"] = (
        "PASS" if unknown_labels_count == 0 else "FAIL"
    )
    evaluations["Statement availability"] = (
        "PASS" if missing_critical_count == 0 else "FAIL"
    )
    evaluations["ID uniqueness"] = (
        "PASS" if duplicate_ids_count == 0 else "FAIL"
    )

    if within_split_duplicate_statements == 0:
        evaluations["Within-split duplicates"] = "PASS"
    elif within_split_duplicate_statements < 50:
        evaluations["Within-split duplicates"] = (
            f"WARNING ({within_split_duplicate_statements} duplicate statements in train)"
        )
    else:
        evaluations["Within-split duplicates"] = "FAIL"

    if cross_split_statement_overlap == 0:
        evaluations["Cross-split leakage risk"] = "PASS"
    elif cross_split_statement_overlap < 20:
        evaluations["Cross-split leakage risk"] = (
            f"WARNING ({cross_split_statement_overlap} cross-split statement overlaps)"
        )
    else:
        evaluations["Cross-split leakage risk"] = "FAIL"

    return evaluations


def run_full_inspection(raw_dir: Path | None = None) -> dict[str, Any]:
    """Run full inspection and print comprehensive formatted report."""
    if raw_dir is None:
        raw_dir = get_project_root() / "data" / "raw"

    print("=" * 80)
    print("LIAR DATASET INSPECTION & VALIDATION REPORT")
    print(f"Target directory: {raw_dir}")
    print("=" * 80)

    # 1. File existence
    print("\n[1] Checking file existence...")
    file_paths = check_files_exist(raw_dir)
    for fname, p in file_paths.items():
        print(f"  [OK] Found {fname} ({p.stat().st_size:,} bytes)")

    # 2. TSV Format & Split Size
    print("\n[2] Loading TSV files and validating 14 columns...")
    splits = load_all_splits(raw_dir)
    total_rows = sum(len(df) for df in splits.values())
    for name, df in splits.items():
        print(
            f"  [OK] {name.capitalize()} split loaded: {len(df):,} rows, {df.shape[1]} columns"
        )
    print(f"  Total dataset rows across all splits: {total_rows:,}")

    # 3. Label Distribution
    print("\n[3] Analyzing label distributions...")
    label_reports: dict[str, Any] = {}
    total_unknown_labels = 0
    for name, df in splits.items():
        rep = inspect_label_distribution(df, name)
        label_reports[name] = rep
        total_unknown_labels += len(rep["unknown_labels"])
        print(f"\n  --- {name.upper()} SPLIT (Total: {rep['total']:,}) ---")
        for lbl in sorted(ALLOWED_LABELS):
            count = rep["counts"].get(lbl, 0)
            pct = rep["percentages"].get(lbl, 0.0)
            print(f"    {lbl:12s}: {count:5d} ({pct:6.2f}%)")
        if rep["unknown_labels"]:
            print(f"    [WARN] UNKNOWN LABELS DETECTED: {rep['unknown_labels']}")
        else:
            print("    [OK] All labels belong to expected 6 benchmark categories.")

    # 4. Missing Values
    print("\n[4] Analyzing missing values per column...")
    missing_reports: dict[str, dict[str, int]] = {}
    total_statement_missing = 0
    total_label_missing = 0
    for name, df in splits.items():
        missing = inspect_missing_values(df)
        missing_reports[name] = missing
        stmt_missing = missing.get("statement", 0)
        lbl_missing = missing.get("label", 0)
        total_statement_missing += stmt_missing
        total_label_missing += lbl_missing
        print(f"\n  --- {name.upper()} MISSING VALUES ---")
        for col, cnt in missing.items():
            if cnt > 0:
                print(f"    {col:18s}: {cnt:5d} missing ({cnt/len(df)*100:.2f}%)")
        print(f"    Statement missing count : {stmt_missing}")
        print(f"    Label missing count     : {lbl_missing}")

    # 5. Empty Statements
    print("\n[5] Checking for empty or whitespace-only statements...")
    empty_reports: dict[str, int] = {}
    total_empty_statements = 0
    for name, df in splits.items():
        empty_cnt = inspect_empty_statements(df)
        empty_reports[name] = empty_cnt
        total_empty_statements += empty_cnt
        print(f"  {name.capitalize()} empty/whitespace statements: {empty_cnt}")

    # 6. Duplicates within each split
    print("\n[6] Checking within-split duplicates...")
    within_split_dups: dict[str, dict[str, int]] = {}
    total_dup_ids = 0
    total_within_split_dup_stmts = 0
    for name, df in splits.items():
        dups = inspect_within_split_duplicates(df)
        within_split_dups[name] = dups
        total_dup_ids += dups["duplicate_ids"]
        total_within_split_dup_stmts += dups["duplicate_statements"]
        print(
            f"  {name.capitalize()}: {dups['duplicate_ids']} duplicate IDs, "
            f"{dups['duplicate_statements']} duplicate statements"
        )

    # 7. Cross-split duplicates
    print("\n[7] Checking cross-split overlap (ID and Statement)...")
    overlap = inspect_cross_split_overlap(splits)
    total_cross_stmt_overlap = sum(
        v["statement_overlap"] for v in overlap.values()
    )
    for pair, counts in overlap.items():
        print(
            f"  {pair}: ID overlap = {counts['id_overlap']}, "
            f"Statement overlap = {counts['statement_overlap']}"
        )

    # 8. Text length statistics
    print("\n[8] Statement text length statistics...")
    length_stats: dict[str, Any] = {}
    for name, df in splits.items():
        stats = inspect_text_lengths(df)
        length_stats[name] = stats
        c_stats = stats["characters"]
        w_stats = stats["words"]
        print(
            f"  {name.capitalize():<6} Chars: min={c_stats['min']:.0f}, med={c_stats['median']:.1f}, "
            f"mean={c_stats['mean']:.2f}, max={c_stats['max']:.0f} | "
            f"Words: min={w_stats['min']:.0f}, med={w_stats['median']:.1f}, "
            f"mean={w_stats['mean']:.2f}, max={w_stats['max']:.0f}"
        )

    # 9. Sample records
    print("\n[9] Sample records from each split (id, label, statement):")
    for name, df in splits.items():
        print(f"\n  --- Sample from {name.upper()} ---")
        for _, row in df.head(2).iterrows():
            stmt_snippet = (
                row["statement"][:75] + "..."
                if len(row["statement"]) > 75
                else row["statement"]
            )
            print(f"    [{row['id']}] ({row['label']}): \"{stmt_snippet}\"")

    # 10. Class balance (Training Split)
    print("\n[10] Training split class balance:")
    train_dist = label_reports["train"]
    for lbl in sorted(ALLOWED_LABELS):
        pct = train_dist["percentages"].get(lbl, 0.0)
        cnt = train_dist["counts"].get(lbl, 0)
        print(f"    {lbl:12s}: {cnt:5d} ({pct:5.2f}%)")

    # 11. Data Quality Checks
    print("\n" + "=" * 80)
    print("BASIC DATA QUALITY CHECK")
    print("=" * 80)
    quality = evaluate_data_quality(
        files_ok=True,
        columns_ok=True,
        unknown_labels_count=total_unknown_labels,
        missing_critical_count=total_statement_missing
        + total_label_missing
        + total_empty_statements,
        duplicate_ids_count=total_dup_ids,
        within_split_duplicate_statements=total_within_split_dup_stmts,
        cross_split_statement_overlap=total_cross_stmt_overlap,
    )
    for cat, status in quality.items():
        print(f"  {cat:<30}: {status}")
    print("=" * 80)

    return {
        "splits": splits,
        "total_rows": total_rows,
        "label_reports": label_reports,
        "missing_reports": missing_reports,
        "empty_reports": empty_reports,
        "within_split_dups": within_split_dups,
        "overlap": overlap,
        "length_stats": length_stats,
        "quality": quality,
    }


if __name__ == "__main__":
    run_full_inspection()
