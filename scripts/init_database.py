"""Production Database Initialization CLI Script (Step 16).

Initializes the empty SQLite production schema at the configured DATABASE_PATH.
Does NOT insert any demo, test, or synthetic rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DATABASE_PATH
from backend.app.db.repository import AnalysisRepository
from backend.app.db.schema import init_database


def main() -> None:
    print("=" * 80)
    print("Production Database Initialization (Step 16)")
    print("=" * 80)
    print(f"Target Database Path : {DATABASE_PATH}")

    # Idempotent schema initialization
    init_database(DATABASE_PATH)
    print("Schema Initialization: SUCCESS (tables: 'analyses', 'feedback'; indexes created)")

    repo = AnalysisRepository(DATABASE_PATH, auto_init=False)
    analysis_count = repo.count_analyses()
    feedback_count = repo.count_feedback()

    print(f"Total Analyses       : {analysis_count}")
    print(f"Total Feedback Items : {feedback_count}")

    if analysis_count == 0 and feedback_count == 0:
        print("Database Status      : CLEAN (Zero demo/synthetic rows present)")
    else:
        print(f"Database Status      : Contains {analysis_count} existing records.")

    print("=" * 80)


if __name__ == "__main__":
    main()
