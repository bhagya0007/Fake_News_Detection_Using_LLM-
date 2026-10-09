"""CLI Entrypoint for Baseline Model Hyperparameter Tuning.

Executes the 96-configuration controlled grid search using
backend.app.services.tune_baseline.run_hyperparameter_search().
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.tune_baseline import run_hyperparameter_search


def main() -> None:
    """Execute hyperparameter tuning."""
    run_hyperparameter_search()


if __name__ == "__main__":
    main()
