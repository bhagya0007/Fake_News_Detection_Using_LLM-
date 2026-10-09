"""Analysis Execution, History, and Detail Endpoints (Step 17).

Provides primary REST endpoints for running multi-signal verification,
retrieving auditable persisted analyses, and managing historical records.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status

from backend.app.api.dependencies import get_analysis_service, get_repository
from backend.app.db.exceptions import AnalysisNotFoundError
from backend.app.db.repository import AnalysisRepository
from backend.app.schemas import (
    AnalysisCreateResponse,
    AnalysisDeleteResponse,
    AnalysisHistoryResponse,
    AnalysisRequest,
    StoredAnalysisResponse,
)
from backend.app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisCreateResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Multi-Signal Analysis and Persist Result",
    description=(
        "Executes the full hybrid verification pipeline (text cleaning, claim extraction, "
        "classical ML classification, stylometric red flags, semantic evidence retrieval, "
        "and evidence-bound reasoning). Persists the resulting analysis into SQLite and "
        "returns the persisted analysis_id along with the complete AnalysisResponse."
    ),
)
def analyze_statement(
    payload: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisCreateResponse:
    """Analyze a claim or article text and persist the result."""
    response = service.analyze_and_save(text=payload.text)
    # response.analysis_id is guaranteed populated by service.analyze_and_save
    return AnalysisCreateResponse(
        analysis_id=str(response.analysis_id),
        analysis=response,
    )


@router.get(
    "/analyses",
    response_model=AnalysisHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List Analysis History (Newest First)",
    description=(
        "Returns a paginated list of lightweight analysis summaries ordered by creation time descending. "
        "Omits heavy nested evidence structures for fast navigation."
    ),
)
def list_analyses(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of items to return."),
    offset: int = Query(default=0, ge=0, description="Number of items to skip."),
    repo: AnalysisRepository = Depends(get_repository),
) -> AnalysisHistoryResponse:
    """Retrieve paginated historical analysis summaries."""
    items = repo.list_analyses(limit=limit, offset=offset)
    total = repo.count_analyses()
    return AnalysisHistoryResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/analyses/{analysis_id}",
    response_model=StoredAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Auditable Persisted Analysis by ID",
    description=(
        "Reconstructs the complete historical AnalysisResponse from SQLite storage without "
        "recomputing models, embeddings, or LLM reasoning."
    ),
)
def get_analysis_detail(
    analysis_id: str,
    repo: AnalysisRepository = Depends(get_repository),
) -> StoredAnalysisResponse:
    """Retrieve full persisted analysis record by ID."""
    record = repo.get_analysis_record(analysis_id)
    if record is None:
        raise AnalysisNotFoundError(f"Analysis with ID '{analysis_id}' was not found.")

    return StoredAnalysisResponse(
        analysis_id=record.analysis_id,
        created_at=record.created_at,
        pipeline_version=record.pipeline_version,
        analysis=record.response,
    )


@router.delete(
    "/analyses/{analysis_id}",
    response_model=AnalysisDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Analysis and Cascade Linked Feedback",
    description="Deletes the specified analysis from SQLite and automatically removes all associated feedback records.",
)
def delete_analysis(
    analysis_id: str,
    repo: AnalysisRepository = Depends(get_repository),
) -> AnalysisDeleteResponse:
    """Delete an analysis and cascaded feedback by ID."""
    deleted = repo.delete_analysis(analysis_id)
    if not deleted:
        raise AnalysisNotFoundError(f"Analysis with ID '{analysis_id}' was not found.")

    return AnalysisDeleteResponse(deleted=True, analysis_id=analysis_id)
