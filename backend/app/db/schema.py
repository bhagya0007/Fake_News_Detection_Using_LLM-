"""Database Schema Definitions and Idempotent Initialization (Step 16).

Defines the relational tables for analyses and user feedback, along with
indexes for temporal ordering and foreign key navigation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from backend.app.db.connection import get_connection

DDL_CREATE_ANALYSES_TABLE = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    cleaned_text TEXT,
    verdict TEXT NOT NULL,
    abstained INTEGER NOT NULL,
    support_score REAL NOT NULL,
    reason TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    ml_json TEXT,
    red_flags_json TEXT,
    llm_json TEXT,
    evidence_json TEXT NOT NULL,
    component_statuses_json TEXT,
    timings_json TEXT,
    analysis_response_json TEXT NOT NULL,
    pipeline_version TEXT NOT NULL
);
"""

DDL_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT,
    corrected_verdict TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
);
"""

DDL_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_analysis_id ON feedback (analysis_id);
"""


def init_database(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize the SQLite database schema idempotently.

    Safe to call repeatedly across application restarts. Uses
    'CREATE TABLE IF NOT EXISTS' and 'CREATE INDEX IF NOT EXISTS'
    to ensure no existing data or schema elements are overwritten or dropped.

    Args:
        db_path: Target database path, or None for configured default.
    """
    with get_connection(db_path) as conn:
        conn.execute(DDL_CREATE_ANALYSES_TABLE)
        conn.execute(DDL_CREATE_FEEDBACK_TABLE)
        conn.executescript(DDL_CREATE_INDEXES)
