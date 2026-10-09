"""FastAPI Dependency Providers and Lifespan State Resolvers (Step 17).

Provides application-scoped singletons for HybridAnalysisEngine, AnalysisRepository,
and AnalysisService to prevent expensive re-instantiation across HTTP requests.
"""

from __future__ import annotations

from fastapi import Request

from backend.app.db.repository import AnalysisRepository
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.hybrid_engine import HybridAnalysisEngine


def get_analysis_service(request: Request) -> AnalysisService:
    """Retrieve the application-scoped AnalysisService instance."""
    service: AnalysisService = request.app.state.analysis_service
    return service


def get_repository(request: Request) -> AnalysisRepository:
    """Retrieve the application-scoped AnalysisRepository instance."""
    repo: AnalysisRepository = request.app.state.repository
    return repo


def get_engine(request: Request) -> HybridAnalysisEngine:
    """Retrieve the application-scoped HybridAnalysisEngine instance."""
    engine: HybridAnalysisEngine = request.app.state.engine
    return engine
