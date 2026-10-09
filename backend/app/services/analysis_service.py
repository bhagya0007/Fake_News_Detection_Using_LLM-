"""High-Level Analysis Service Orchestration (Step 16).

Composes the compute engine (HybridAnalysisEngine) and the persistence layer
(AnalysisRepository). Maintains strict separation of concerns: HybridAnalysisEngine
remains independently usable without requiring a database, and database persistence
failures never mutate computed analysis verdicts.
"""

from __future__ import annotations

from typing import Optional

from backend.app.db.exceptions import DatabaseError
from backend.app.db.repository import AnalysisRepository
from backend.app.schemas import (
    AnalysisHistoryItem,
    AnalysisResponse,
    FeedbackRecord,
    FeedbackRequest,
)
from backend.app.services.hybrid_engine import HybridAnalysisEngine


class AnalysisService:
    """Service coordinating end-to-end multi-signal analysis and audit persistence."""

    def __init__(
        self,
        engine: Optional[HybridAnalysisEngine] = None,
        repository: Optional[AnalysisRepository] = None,
    ) -> None:
        """Initialize service with optional engine and repository injections.

        Args:
            engine: HybridAnalysisEngine instance (defaults to new production instance).
            repository: AnalysisRepository instance (defaults to new production repository).
        """
        self.engine = engine if engine is not None else HybridAnalysisEngine()
        self.repository = repository if repository is not None else AnalysisRepository()

    def analyze_and_save(
        self,
        text: str,
        cleaned_text: Optional[str] = None,
    ) -> AnalysisResponse:
        """Execute end-to-end analysis and persist the validated result.

        Architectural Principle:
        - The analysis pipeline runs first in full isolation.
        - Persistence is an orthogonal audit/storage step.
        - If the database write encounters an error, a controlled DatabaseError is raised;
          the underlying analysis verdict is NEVER modified or degraded to unverifiable.

        Args:
            text: Raw input claim or article text.
            cleaned_text: Optional pre-cleaned text override.

        Returns:
            Computed and persisted AnalysisResponse.

        Raises:
            DatabaseError: If database persistence fails after successful analysis computation.
        """
        # 1. Compute multi-signal analysis
        response = self.engine.analyze_text(text)

        # 2. Persist result
        try:
            saved_id = self.repository.save_analysis(
                response=response,
                original_text=text,
                cleaned_text=cleaned_text,
            )
            response.analysis_id = saved_id
        except Exception as exc:
            # Under no circumstances should database failure alter the analysis verdict!
            raise DatabaseError(
                f"Analysis completed with verdict '{response.verdict.value}', but failed to persist: {exc}"
            ) from exc

        return response

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResponse]:
        """Retrieve a previously analyzed and stored report by ID."""
        return self.repository.get_analysis(analysis_id)

    def list_history(self, limit: int = 20, offset: int = 0) -> list[AnalysisHistoryItem]:
        """List historical analyses ordered newest first."""
        return self.repository.list_analyses(limit=limit, offset=offset)

    def submit_feedback(
        self,
        feedback: FeedbackRequest,
        analysis_id: Optional[str] = None,
    ) -> FeedbackRecord:
        """Attach user feedback to an existing stored analysis."""
        return self.repository.save_feedback(feedback, analysis_id=analysis_id)

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis and cascade delete attached feedback."""
        return self.repository.delete_analysis(analysis_id)
