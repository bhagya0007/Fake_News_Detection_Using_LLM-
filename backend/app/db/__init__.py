"""Database and persistence package for Fake News Detection pipeline (Step 16).

Provides SQLite connection lifecycle, schema initialization, and repository operations.
"""

from backend.app.db.connection import get_connection, resolve_db_path
from backend.app.db.exceptions import (
    AnalysisNotFoundError,
    DatabaseError,
    DatabaseInitializationError,
    DataIntegrityError,
    FeedbackValidationError,
)
from backend.app.db.repository import AnalysisRepository
from backend.app.db.schema import init_database

__all__ = [
    "AnalysisNotFoundError",
    "AnalysisRepository",
    "DataIntegrityError",
    "DatabaseError",
    "DatabaseInitializationError",
    "FeedbackValidationError",
    "get_connection",
    "init_database",
    "resolve_db_path",
]
