"""Descriptive Training-Set Stylometric Diagnostic (Step 8).

Analyzes the LIAR training split (data/raw/train.tsv) using the rule-based red-flag detector.
Strictly descriptive analysis — NOT supervised tuning or accuracy measurement.

Rules are designed for human interpretability, not label fitting.
Test data is strictly untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.baseline_data import prepare_baseline_split
from backend.app.services.dataset_loader import get_project_root
from backend.app.services.red_flags import analyze_red_flags


def run_training_diagnostic() -> dict:
    """Analyze the LIAR training split with the red-flag detector."""
    print("=" * 80)
    print("LIAR TRAINING SPLIT RED-FLAG DIAGNOSTIC (DESCRIPTIVE ONLY)")
    print("=" * 80)

    project_root = get_project_root()
    train_df = prepare_baseline_split("train")
    total_rows = len(train_df)
    print(f"Loaded train split: {total_rows:,} statements.")

    start_time = time.perf_counter()

    flag_counts: dict[str, int] = {}
    total_flags_emitted = 0
    high_sev_total = 0
    med_sev_total = 0
    low_sev_total = 0

    statements_with_caps = 0
    statements_with_sensational = 0
    statements_with_stats = 0
    statements_with_attribution = 0
    statements_with_any_flag = 0

    # Per-label tracking for descriptive observation
    label_flag_counts: dict[str, list[int]] = {
        "likely_reliable": [],
        "misleading": [],
        "likely_fake": [],
    }

    for _, row in train_df.iterrows():
        statement = str(row["statement"])
        label = str(row["project_label"])

        res = analyze_red_flags(statement)
        flags = res["flags"]
        feats = res["features"]
        summ = res["summary"]

        n_flags = len(flags)
        total_flags_emitted += n_flags
        if n_flags > 0:
            statements_with_any_flag += 1

        if label in label_flag_counts:
            label_flag_counts[label].append(n_flags)

        high_sev_total += summ["high_severity_flags"]
        med_sev_total += summ["medium_severity_flags"]
        low_sev_total += summ["low_severity_flags"]

        types_in_row = {f["type"] for f in flags}
        for t in types_in_row:
            flag_counts[t] = flag_counts.get(t, 0) + 1

        if "excessive_caps" in types_in_row:
            statements_with_caps += 1
        if "sensational_language" in types_in_row:
            statements_with_sensational += 1
        if summ["has_statistical_claim"] or feats["statistical_expression_count"] > 0:
            statements_with_stats += 1
        if summ["has_attribution"]:
            statements_with_attribution += 1

    duration = time.perf_counter() - start_time
    avg_flags = round(total_flags_emitted / total_rows, 3)

    summary_data = {
        "dataset": "LIAR",
        "split": "train",
        "total_statements_analyzed": total_rows,
        "analysis_duration_seconds": round(duration, 3),
        "throughput_statements_per_sec": round(total_rows / duration, 1),
        "overall_signals": {
            "statements_with_any_flag_pct": round((statements_with_any_flag / total_rows) * 100, 2),
            "excessive_caps_pct": round((statements_with_caps / total_rows) * 100, 2),
            "sensational_language_pct": round((statements_with_sensational / total_rows) * 100, 2),
            "statistical_claim_pct": round((statements_with_stats / total_rows) * 100, 2),
            "attribution_present_pct": round((statements_with_attribution / total_rows) * 100, 2),
            "average_flags_per_statement": avg_flags,
        },
        "severity_totals": {
            "high": high_sev_total,
            "medium": med_sev_total,
            "low": low_sev_total,
            "total_emitted": total_flags_emitted,
        },
        "flag_frequency_breakdown": {
            k: {
                "count": v,
                "percentage": round((v / total_rows) * 100, 2),
            }
            for k, v in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
        },
        "descriptive_label_breakdown": {
            lbl: {
                "sample_count": len(counts),
                "average_flags_per_statement": round(sum(counts) / len(counts), 3) if counts else 0.0,
            }
            for lbl, counts in label_flag_counts.items()
        },
        "methodological_note": (
            "Descriptive stylometric frequency diagnostic only. Red flags are presentation cues, "
            "not proof of falsity. Rules were not tuned or optimized against dataset labels."
        ),
    }

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_file = models_dir / "red_flag_train_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nCompleted in {duration:.2f} seconds ({total_rows/duration:.1f} statements/sec).")
    print(f"Saved diagnostic summary to: {out_file}")
    print("\nKey Training Split Findings:")
    print(f"  - Statements with any flag       : {summary_data['overall_signals']['statements_with_any_flag_pct']}%")
    print(f"  - Statistical assertions present : {summary_data['overall_signals']['statistical_claim_pct']}%")
    print(f"  - Attribution phrases present    : {summary_data['overall_signals']['attribution_present_pct']}%")
    print(f"  - Sensational phrasing detected  : {summary_data['overall_signals']['sensational_language_pct']}%")
    print(f"  - Excessive caps detected        : {summary_data['overall_signals']['excessive_caps_pct']}%")
    print(f"  - Average flags per statement    : {avg_flags}")
    print("=" * 80)

    return summary_data


if __name__ == "__main__":
    run_training_diagnostic()
