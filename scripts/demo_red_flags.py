"""Demonstration Script for Interpretable Red-Flag / Stylometric Detector (Step 8).

Runs the rule-based red-flag analyzer on representative sample statements to display:
- Raw input statement
- Extracted quantitative stylometric features
- Triggered linguistic warning flags with severity and snippets
- High-level flag summary

IMPORTANT:
This component does NOT emit 'fake', 'real', or any truthfulness verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.red_flags import analyze_red_flags


DEMO_STATEMENTS = [
    {
        "category": "Neutral / Factual Report with Attribution",
        "text": "According to the health ministry report, vaccination coverage increased by 4%.",
    },
    {
        "category": "High Sensationalism & Punctuation",
        "text": "SHOCKING!!! You won't believe what doctors discovered!!!",
    },
    {
        "category": "Unsupported Quantitative Statistic",
        "text": "87% of people will suffer from this condition.",
    },
    {
        "category": "Urgency & Call-to-Action",
        "text": "Share this before they delete it!",
    },
    {
        "category": "Mixed Style (Breaking News with Official Attribution)",
        "text": "BREAKING: Officials announced a 12% increase in exports.",
    },
]


def run_demo() -> None:
    """Run red-flag analysis on demonstration statements."""
    print("=" * 80)
    print("INTERPRETABLE RED-FLAG / STYLOMETRIC DETECTOR DEMO (STEP 8)")
    print("=" * 80)
    print("Notice: Red flags are linguistic presentation cues, NOT proof of falsity.")
    print("No truthfulness verdicts ('likely_fake', 'likely_reliable') are generated.\n")

    for idx, item in enumerate(DEMO_STATEMENTS, 1):
        category = item["category"]
        text = item["text"]
        result = analyze_red_flags(text)

        print(f"[{idx}] Category: {category}")
        print(f"    Input Statement: \"{text}\"")
        print("\n    Key Stylometric Features:")
        feats = result["features"]
        print(f"      - Word Count             : {feats['word_count']}")
        print(f"      - Uppercase Ratio        : {feats['uppercase_letter_ratio']:.2%}")
        print(f"      - Exclamation Count      : {feats['exclamation_count']}")
        print(f"      - Question Mark Count    : {feats['question_mark_count']}")
        print(f"      - Percentages Count      : {feats['percentage_count']}")
        print(f"      - Sensational Phrases    : {feats['sensational_phrase_count']}")
        print(f"      - Urgency Phrases        : {feats['urgency_phrase_count']}")
        print(f"      - Attribution Phrases    : {feats['attribution_phrase_count']}")

        print("\n    Triggered Warning Flags:")
        flags = result["flags"]
        if not flags:
            print("      (None - No warning flags triggered)")
        else:
            for f in flags:
                print(f"      * [{f['severity'].upper()}] {f['type']}")
                print(f"        Message: {f['message']}")
                print(f"        Snippet: \"{f['snippet']}\" (value: {f['value']})")

        summary = result["summary"]
        print("\n    Summary:")
        print(f"      Total Flags: {summary['total_flags']} "
              f"(High: {summary['high_severity_flags']}, "
              f"Medium: {summary['medium_severity_flags']}, "
              f"Low: {summary['low_severity_flags']})")
        print(f"      Attribution Present: {summary['has_attribution']}")
        print(f"      Statistical Claim  : {summary['has_statistical_claim']}")
        print("-" * 80)


if __name__ == "__main__":
    run_demo()
