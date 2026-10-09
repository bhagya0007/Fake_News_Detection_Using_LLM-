"""CLI Entrypoint for One-Time Official Test Evaluation of Frozen Classical Model (Step 7).

Evaluates models/classical_best_validation.joblib on data/raw/test.tsv.
ZERO FITTING IS PERFORMED.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.evaluate_classical import evaluate_frozen_classical_model


def main() -> None:
    """Execute official test evaluation."""
    evaluate_frozen_classical_model()


if __name__ == "__main__":
    main()
