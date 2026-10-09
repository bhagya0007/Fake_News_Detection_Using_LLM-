"""CLI Entrypoint for Baseline Model Training and Evaluation.

Invokes the reusable training service in backend.app.services.train_baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.train_baseline import train_and_evaluate_baseline


def main() -> None:
    """Execute training script."""
    train_and_evaluate_baseline()


if __name__ == "__main__":
    main()
