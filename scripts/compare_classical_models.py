"""CLI Entrypoint for Classical Model Comparison (Step 6.5).

Compares the current tuned Logistic Regression baseline against:
1. LinearSVC with word TF-IDF (Model B)
2. Word + character TF-IDF with Logistic Regression (Model C)
3. Word + character TF-IDF with LinearSVC (Model D)

All fitting is performed ONLY on train.tsv.
All comparison and selection is performed ONLY on valid.tsv.
The official test split (data/raw/test.tsv) is strictly untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.compare_classical import run_classical_comparison


def main() -> None:
    """Execute classical model comparison."""
    run_classical_comparison()


if __name__ == "__main__":
    main()
