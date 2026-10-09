"""Persistence and Database Exception Hierarchy (Step 16).

Encapsulates all database-related error conditions to prevent raw SQL errors
or internal database paths from leaking to upstream layers.
"""

from __future__ import annotations


class DatabaseError(Exception):
    """Base exception for all database and persistence operations."""


class DatabaseInitializationError(DatabaseError):
    """Raised when database directory creation or schema initialization fails."""


class AnalysisNotFoundError(DatabaseError):
    """Raised when an analysis with the specified ID does not exist."""


class FeedbackValidationError(DatabaseError):
    """Raised when feedback input fails domain validation before persistence."""


class DataIntegrityError(DatabaseError):
    """Raised when persisted JSON cannot be validated against Pydantic schema contracts."""
