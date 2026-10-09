"""Operational Health and Liveness Route (Step 17).

Provides ultra-lightweight liveness checks for load balancers, orchestrators,
and uptime probes. Strictly zero LLM or neural model invocations.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from backend.app.config import LLM_PROVIDER, PIPELINE_VERSION
from backend.app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Operational Liveness Probe",
    description="Returns high-level service status without triggering external AI or database queries.",
)
@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,  # Alias route
)
def get_health() -> HealthResponse:
    """Return service status and version information (constant-time response)."""
    return HealthResponse(
        status="ok",
        service="fake-news-detection-api",
        pipeline_version=PIPELINE_VERSION,
        llm_provider=LLM_PROVIDER or "mock",
    )
