"""Centralized API Error Handlers and Exception Mapping (Step 17).

Translates internal domain exceptions into clean, structured HTTP error responses
without leaking internal stack traces, SQL syntax, or authentication credentials.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app.db.exceptions import (
    AnalysisNotFoundError,
    DatabaseError,
    DataIntegrityError,
    FeedbackValidationError,
)

logger = logging.getLogger("backend.app.api.errors")


def register_error_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI application."""

    @app.exception_handler(AnalysisNotFoundError)
    async def handle_not_found(_: Request, exc: AnalysisNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "not_found",
                "message": str(exc),
                "detail": None,
            },
        )

    @app.exception_handler(FeedbackValidationError)
    async def handle_feedback_validation(_: Request, exc: FeedbackValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": str(exc),
                "detail": None,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Extract sanitized validation failure messages
        errors = exc.errors()
        first_msg = errors[0]["msg"] if errors else "Invalid request payload."
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": f"Input validation failed: {first_msg}",
                "detail": str(errors[:3]),  # Bounded summary, zero secrets
            },
        )

    @app.exception_handler(ValidationError)
    async def handle_pydantic_validation(_: Request, exc: ValidationError) -> JSONResponse:
        errors = exc.errors()
        first_msg = errors[0]["msg"] if errors else "Invalid data structure."
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": f"Contract validation failed: {first_msg}",
                "detail": None,
            },
        )

    @app.exception_handler(DatabaseError)
    async def handle_database_error(_: Request, exc: DatabaseError) -> JSONResponse:
        logger.error(f"Persistence error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "service_unavailable",
                "message": "Persistence service unavailable or failed to persist record.",
                "detail": None,
            },
        )

    @app.exception_handler(DataIntegrityError)
    async def handle_data_integrity(_: Request, exc: DataIntegrityError) -> JSONResponse:
        logger.error(f"Data integrity error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "data_integrity_error",
                "message": "Stored analysis record could not be reconstructed.",
                "detail": None,
            },
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": str(exc.detail),
                "detail": None,
            },
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(_: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected server error occurred during processing.",
                "detail": None,
            },
        )
