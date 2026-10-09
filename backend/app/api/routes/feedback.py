"""User Feedback Collection and Audit Retrieval Endpoints (Step 17).

Provides REST endpoints to submit qualitative feedback or verdict corrections
for an existing analysis. Notice: Feedback is audit data only and never triggers
automated retraining of classical or neural models.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from backend.app.api.dependencies import get_repository
from backend.app.db.exceptions import AnalysisNotFoundError
from backend.app.db.repository import AnalysisRepository
from backend.app.schemas import FeedbackRecord, FeedbackRequest

router = APIRouter(prefix="/api/v1/analyses/{analysis_id}", tags=["Feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Submit User Feedback on an Analysis",
    description=(
        "Attaches qualitative rating, comments, or suggested verdict corrections to a "
        "previously stored analysis. Feedback is retained strictly for offline inspection "
        "and evaluation benchmarks; it never triggers automatic online retraining."
    ),
)
def submit_feedback(
    analysis_id: str,
    feedback: FeedbackRequest,
    repo: AnalysisRepository = Depends(get_repository),
) -> FeedbackRecord:
    """Submit feedback for a specific analysis."""
    # Ensure feedback payload analysis_id matches path parameter
    return repo.save_feedback(feedback=feedback, analysis_id=analysis_id)


@router.get(
    "/feedback",
    response_model=list[FeedbackRecord],
    status_code=status.HTTP_200_OK,
    summary="List Feedback Attached to an Analysis",
    description="Returns all feedback records submitted for the specified analysis ID, ordered chronologically.",
)
def list_feedback(
    analysis_id: str,
    repo: AnalysisRepository = Depends(get_repository),
) -> list[FeedbackRecord]:
    """Retrieve all feedback attached to the target analysis."""
    # Enforce parent existence check: nonexistent analysis returns 404
    if repo.get_analysis(analysis_id) is None:
        raise AnalysisNotFoundError(f"Analysis with ID '{analysis_id}' was not found.")

    return repo.get_feedback_for_analysis(analysis_id)
