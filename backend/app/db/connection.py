"""SQLite Connection Management and Lifecycle (Step 16).

Provides safe, context-managed SQLite connections with foreign keys enabled,
WAL journal mode, Row factories, and explicit transaction management.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from backend.app.config import DATABASE_PATH
from backend.app.db.exceptions import DatabaseInitializationError


def resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve database path and ensure its parent directory exists.

    Args:
        db_path: Custom SQLite database path, or None to use default DATABASE_PATH.

    Returns:
        Resolved absolute Path instance (or Path(':memory:') for in-memory DBs).
    """
    if db_path is not None and str(db_path) == ":memory:":
        return Path(":memory:")

    path = Path(db_path).resolve() if db_path is not None else DATABASE_PATH.resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseInitializationError(
            f"Failed to create database directory '{path.parent}': {exc}"
        ) from exc
    return path


@contextmanager
def get_connection(
    db_path: Optional[Union[str, Path]] = None,
    timeout: float = 30.0,
    enable_wal: bool = True,
) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite connection as a context manager.

    Features:
    - Row factory configured to sqlite3.Row for dict-like column access.
    - Foreign keys explicitly enabled (`PRAGMA foreign_keys = ON`).
    - WAL journaling enabled where supported (`PRAGMA journal_mode = WAL`).
    - Automatic commit on successful block exit.
    - Automatic rollback on unhandled exceptions.
    - Guaranteed connection closure in finally block.

    Args:
        db_path: Target database path, or None for configured default.
        timeout: Lock acquisition timeout in seconds.
        enable_wal: Whether to attempt enabling Write-Ahead Logging.

    Yields:
        Configured sqlite3.Connection instance.
    """
    resolved_path = resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved_path), timeout=timeout)
    conn.row_factory = sqlite3.Row

    try:
        # Mandatory SQLite integrity settings
        conn.execute("PRAGMA foreign_keys = ON;")

        if enable_wal and str(resolved_path) != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
            except sqlite3.OperationalError:
                # Fall back gracefully if filesystem prohibits WAL
                pass

        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
