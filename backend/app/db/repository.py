"""Analysis and Feedback Data Repository (Step 16).

Provides parameterized CRUD operations for analyses and user feedback,
structured JSON serialization, and validated round-trip reconstruction.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from backend.app.config import PIPELINE_VERSION
from backend.app.db.connection import get_connection, resolve_db_path
from backend.app.db.exceptions import (
    AnalysisNotFoundError,
    DatabaseError,
    DataIntegrityError,
    FeedbackValidationError,
)
from backend.app.db.schema import init_database
from backend.app.schemas import (
    AnalysisHistoryItem,
    AnalysisRecord,
    AnalysisResponse,
    FeedbackRating,
    FeedbackRecord,
    FeedbackRequest,
    VerdictLabel,
)


class AnalysisRepository:
    """Repository handling persistence, retrieval, and audit queries for analyses."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        pipeline_version: Optional[str] = None,
        auto_init: bool = True,
    ) -> None:
        """Initialize repository with target database path and pipeline version.

        Args:
            db_path: Path to SQLite database file, or None for configured default.
            pipeline_version: Audit pipeline version tag (defaults to config PIPELINE_VERSION).
            auto_init: If True, executes idempotent schema initialization on creation.
        """
        self.db_path = resolve_db_path(db_path)
        self.pipeline_version = pipeline_version or PIPELINE_VERSION
        if auto_init:
            init_database(self.db_path)

    def save_analysis(
        self,
        response: AnalysisResponse,
        original_text: Optional[str] = None,
        cleaned_text: Optional[str] = None,
    ) -> str:
        """Save a completed AnalysisResponse to the database.

        Generates a collision-resistant UUID4 if not already present on the response,
        records timezone-aware UTC creation time, and serializes the complete response
        as structured JSON for future reconstruction without recomputation.

        Args:
            response: Completed AnalysisResponse model.
            original_text: Raw input text before any cleaning (defaults to response.input_text).
            cleaned_text: Text after preprocessing (optional).

        Returns:
            The saved analysis_id string.
        """
        # Assign UUID4 if absent
        if not response.analysis_id or not str(response.analysis_id).strip():
            analysis_id = str(uuid.uuid4())
            response.analysis_id = analysis_id
        else:
            analysis_id = str(response.analysis_id)

        # Ensure timezone-aware UTC timestamp
        if response.created_at is None:
            created_at_dt = datetime.now(timezone.utc)
            response.created_at = created_at_dt
        else:
            created_at_dt = (
                response.created_at
                if response.created_at.tzinfo is not None
                else response.created_at.replace(tzinfo=timezone.utc)
            )

        input_text = original_text if original_text is not None else response.input_text
        abstained_int = 1 if response.verdict == VerdictLabel.unverifiable else 0

        # Serialize component structures safely via Pydantic model_dump(mode="json")
        try:
            claims_json = json.dumps([c.model_dump(mode="json") for c in response.claims])
            ml_json = (
                json.dumps(response.ml.model_dump(mode="json"))
                if response.ml is not None
                else None
            )
            red_flags_json = (
                json.dumps(response.red_flags.model_dump(mode="json"))
                if response.red_flags is not None
                else None
            )
            llm_json = (
                json.dumps(response.llm.model_dump(mode="json"))
                if response.llm is not None
                else None
            )
            evidence_json = json.dumps([e.model_dump(mode="json") for e in response.evidence])
            component_statuses_json = (
                json.dumps([cs.model_dump(mode="json") for cs in response.component_statuses])
                if response.component_statuses
                else None
            )
            timings_json = (
                json.dumps(response.timings.model_dump(mode="json"))
                if response.timings is not None
                else None
            )
            analysis_response_json = json.dumps(response.model_dump(mode="json"))
        except Exception as exc:
            raise DatabaseError(f"Failed to serialize AnalysisResponse to JSON: {exc}") from exc

        query = """
        INSERT INTO analyses (
            analysis_id, created_at, input_text, cleaned_text, verdict, abstained,
            support_score, reason, claims_json, ml_json, red_flags_json, llm_json,
            evidence_json, component_statuses_json, timings_json,
            analysis_response_json, pipeline_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            analysis_id,
            created_at_dt.isoformat(),
            input_text,
            cleaned_text,
            response.verdict.value,
            abstained_int,
            float(response.support_score),
            response.reason,
            claims_json,
            ml_json,
            red_flags_json,
            llm_json,
            evidence_json,
            component_statuses_json,
            timings_json,
            analysis_response_json,
            self.pipeline_version,
        )

        try:
            with get_connection(self.db_path) as conn:
                conn.execute(query, params)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database error while saving analysis '{analysis_id}': {exc}") from exc

        return analysis_id

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResponse]:
        """Retrieve and reconstruct a complete AnalysisResponse by its ID.

        Validates the stored JSON against the AnalysisResponse Pydantic contract.

        Args:
            analysis_id: Collision-resistant analysis UUID string.

        Returns:
            Reconstructed AnalysisResponse, or None if not found.

        Raises:
            DataIntegrityError: If stored record cannot be deserialized or validated.
        """
        if not analysis_id or not str(analysis_id).strip():
            return None

        query = "SELECT analysis_response_json FROM analyses WHERE analysis_id = ?"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (str(analysis_id).strip(),))
            row = cursor.fetchone()

        if row is None:
            return None

        raw_json = row["analysis_response_json"]
        try:
            data = json.loads(raw_json)
            return AnalysisResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            raise DataIntegrityError(
                f"Failed to reconstruct AnalysisResponse for ID '{analysis_id}': {exc}"
            ) from exc

    def get_analysis_record(self, analysis_id: str) -> Optional[AnalysisRecord]:
        """Retrieve full database record including audit metadata and child response.

        Args:
            analysis_id: Analysis identifier.

        Returns:
            Populated AnalysisRecord, or None if not found.
        """
        if not analysis_id or not str(analysis_id).strip():
            return None

        query = "SELECT * FROM analyses WHERE analysis_id = ?"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (str(analysis_id).strip(),))
            row = cursor.fetchone()

        if row is None:
            return None

        try:
            raw_response = json.loads(row["analysis_response_json"])
            response_obj = AnalysisResponse.model_validate(raw_response)
            created_at_dt = datetime.fromisoformat(row["created_at"])
            if created_at_dt.tzinfo is None:
                created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

            return AnalysisRecord(
                analysis_id=row["analysis_id"],
                created_at=created_at_dt,
                input_text=row["input_text"],
                cleaned_text=row["cleaned_text"],
                verdict=VerdictLabel(row["verdict"]),
                abstained=bool(row["abstained"]),
                support_score=float(row["support_score"]),
                reason=row["reason"],
                pipeline_version=row["pipeline_version"],
                response=response_obj,
            )
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            raise DataIntegrityError(
                f"Data integrity error in record for ID '{analysis_id}': {exc}"
            ) from exc

    def list_analyses(self, limit: int = 20, offset: int = 0) -> list[AnalysisHistoryItem]:
        """List lightweight analysis summaries ordered by creation date descending.

        Args:
            limit: Maximum items to return (1 <= limit <= 100).
            offset: Zero-based offset index (offset >= 0).

        Returns:
            List of AnalysisHistoryItem summaries.
        """
        if limit < 1 or limit > 100:
            raise ValueError(f"Limit must be between 1 and 100 (got {limit}).")
        if offset < 0:
            raise ValueError(f"Offset must be non-negative (got {offset}).")

        query = """
        SELECT analysis_id, created_at, input_text, verdict, abstained, support_score, pipeline_version
        FROM analyses
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (limit, offset))
            rows = cursor.fetchall()

        items: list[AnalysisHistoryItem] = []
        for r in rows:
            text = r["input_text"]
            preview = text[:150] + ("..." if len(text) > 150 else "")
            dt = datetime.fromisoformat(r["created_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            items.append(
                AnalysisHistoryItem(
                    analysis_id=r["analysis_id"],
                    created_at=dt,
                    input_preview=preview,
                    verdict=VerdictLabel(r["verdict"]),
                    abstained=bool(r["abstained"]),
                    support_score=float(r["support_score"]),
                    pipeline_version=r["pipeline_version"],
                )
            )
        return items

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis and cascade deletion to associated feedback rows.

        Args:
            analysis_id: Target analysis ID to delete.

        Returns:
            True if row was found and deleted, False otherwise.
        """
        if not analysis_id or not str(analysis_id).strip():
            return False

        query = "DELETE FROM analyses WHERE analysis_id = ?"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (str(analysis_id).strip(),))
            return cursor.rowcount > 0

    def save_feedback(
        self,
        feedback: FeedbackRequest,
        analysis_id: Optional[str] = None,
    ) -> FeedbackRecord:
        """Store optional user feedback linked to an existing analysis.

        Notice: Feedback is stored strictly for audit and post-hoc research.
        It is never used to automatically retrain models or modify fusion weights.

        Args:
            feedback: Validated FeedbackRequest payload.
            analysis_id: Optional override analysis ID (defaults to feedback.analysis_id).

        Returns:
            Persisted FeedbackRecord model.

        Raises:
            AnalysisNotFoundError: If target analysis ID does not exist in analyses table.
            FeedbackValidationError: If rating or corrected_verdict values are invalid.
        """
        target_id = str(analysis_id or feedback.analysis_id).strip()
        if not target_id:
            raise AnalysisNotFoundError("An analysis_id must be provided to attach feedback.")

        # Verify parent analysis exists (enforcing relational integrity)
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM analyses WHERE analysis_id = ?", (target_id,))
            if cursor.fetchone() is None:
                raise AnalysisNotFoundError(f"Analysis with ID '{target_id}' does not exist.")

        # Validate rating
        rating_val = (
            feedback.rating.value
            if isinstance(feedback.rating, FeedbackRating)
            else str(feedback.rating)
        )
        if rating_val not in {"helpful", "not_helpful", "unsure"}:
            raise FeedbackValidationError(
                f"Invalid rating '{rating_val}'. Allowed values: helpful, not_helpful, unsure."
            )

        # Validate corrected verdict if provided
        corrected_val: Optional[str] = None
        if feedback.corrected_verdict is not None:
            corrected_val = (
                feedback.corrected_verdict.value
                if isinstance(feedback.corrected_verdict, VerdictLabel)
                else str(feedback.corrected_verdict)
            )
            if corrected_val not in {"likely_reliable", "misleading", "likely_fake", "unverifiable"}:
                raise FeedbackValidationError(f"Invalid corrected_verdict '{corrected_val}'.")

        feedback_id = str(uuid.uuid4())
        created_at_dt = datetime.now(timezone.utc)

        query = """
        INSERT INTO feedback (
            feedback_id, analysis_id, created_at, rating, comment, corrected_verdict
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            feedback_id,
            target_id,
            created_at_dt.isoformat(),
            rating_val,
            feedback.comment,
            corrected_val,
        )

        with get_connection(self.db_path) as conn:
            conn.execute(query, params)

        return FeedbackRecord(
            feedback_id=feedback_id,
            analysis_id=target_id,
            created_at=created_at_dt,
            rating=FeedbackRating(rating_val),
            comment=feedback.comment,
            corrected_verdict=VerdictLabel(corrected_val) if corrected_val else None,
        )

    def get_feedback_for_analysis(self, analysis_id: str) -> list[FeedbackRecord]:
        """Retrieve all feedback records attached to a specific analysis.

        Args:
            analysis_id: Parent analysis ID.

        Returns:
            List of FeedbackRecord items ordered by creation time ascending.
        """
        if not analysis_id or not str(analysis_id).strip():
            return []

        query = """
        SELECT feedback_id, analysis_id, created_at, rating, comment, corrected_verdict
        FROM feedback
        WHERE analysis_id = ?
        ORDER BY created_at ASC
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (str(analysis_id).strip(),))
            rows = cursor.fetchall()

        records: list[FeedbackRecord] = []
        for r in rows:
            dt = datetime.fromisoformat(r["created_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            cv = r["corrected_verdict"]
            records.append(
                FeedbackRecord(
                    feedback_id=r["feedback_id"],
                    analysis_id=r["analysis_id"],
                    created_at=dt,
                    rating=FeedbackRating(r["rating"]),
                    comment=r["comment"],
                    corrected_verdict=VerdictLabel(cv) if cv else None,
                )
            )
        return records

    def count_analyses(self) -> int:
        """Return the total number of saved analyses."""
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) AS total FROM analyses")
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def count_feedback(self) -> int:
        """Return the total number of saved feedback records."""
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) AS total FROM feedback")
            row = cursor.fetchone()
            return int(row["total"]) if row else 0
