"""Main FastAPI Application Entry Point and App Factory (Step 17).

Provides a modular FastAPI application with clean lifespan state initialization,
dependency injection overrides, centralized error handling, and structured routing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI

from backend.app.api.errors import register_error_handlers
from backend.app.api.routes import analysis_router, feedback_router, health_router
from backend.app.config import PROJECT_NAME, VERSION
from backend.app.db.repository import AnalysisRepository
from backend.app.db.schema import init_database
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.hybrid_engine import HybridAnalysisEngine

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Liveness probes and operational status indicators.",
    },
    {
        "name": "Analysis",
        "description": "Multi-signal fact-checking verification, history, and detail retrieval.",
    },
    {
        "name": "Feedback",
        "description": "Qualitative feedback submission and post-hoc evaluation records.",
    },
]


def create_app(
    analysis_service: Optional[AnalysisService] = None,
    repository: Optional[AnalysisRepository] = None,
    engine: Optional[HybridAnalysisEngine] = None,
    init_db: bool = True,
) -> FastAPI:
    """Create and configure a production-style FastAPI application.

    Supports optional dependency injection for testing and demonstration contexts.

    Args:
        analysis_service: Pre-configured AnalysisService, or None to build from defaults.
        repository: Pre-configured AnalysisRepository, or None to use default database.
        engine: Pre-configured HybridAnalysisEngine, or None to build production engine.
        init_db: Whether to idempotently ensure database schema on startup.

    Returns:
        Configured FastAPI application instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Lifespan hook ensures singletons are initialized if not already provided
        if not hasattr(app.state, "repository") or app.state.repository is None:
            app.state.repository = repository if repository is not None else AnalysisRepository(auto_init=False)
        if init_db:
            init_database(app.state.repository.db_path)

        if not hasattr(app.state, "engine") or app.state.engine is None:
            app.state.engine = engine if engine is not None else HybridAnalysisEngine(allow_synthetic_evidence=False)

        if not hasattr(app.state, "analysis_service") or app.state.analysis_service is None:
            app.state.analysis_service = (
                analysis_service
                if analysis_service is not None
                else AnalysisService(engine=app.state.engine, repository=app.state.repository)
            )

        yield

    app = FastAPI(
        title="Fake News Detection Using LLMs API",
        description="Hybrid evidence-aware decision-support API for factual claim analysis.",
        version=VERSION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Immediately populate state if provided (for testing and non-lifespan contexts)
    if repository is not None:
        app.state.repository = repository
        if init_db:
            init_database(repository.db_path)
    if engine is not None:
        app.state.engine = engine
    if analysis_service is not None:
        app.state.analysis_service = analysis_service
    elif repository is not None and engine is not None:
        app.state.analysis_service = AnalysisService(engine=engine, repository=repository)

    # Register centralized exception handlers
    register_error_handlers(app)

    # Register modular route collections
    app.include_router(health_router)
    app.include_router(analysis_router)
    app.include_router(feedback_router)

    return app


# Default application instance for ASGI servers (uvicorn backend.app.api.app:app)
app = create_app()
