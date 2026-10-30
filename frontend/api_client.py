"""HTTP Client for Fake News Detection FastAPI Backend (Step 18).

Provides structured HTTP communication between the Streamlit frontend and the
FastAPI backend service. Strictly isolates the frontend from ML models, SQLite,
and direct LLM SDKs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

# Shared constant matching backend validation bounds
MAX_ANALYSIS_TEXT_LENGTH: int = 20_000
DEFAULT_API_BASE_URL: str = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS: float = 180.0


class APIClientError(Exception):
    """Base exception for all API client errors."""


class APIConnectionError(APIClientError):
    """Raised when the backend API server is unreachable."""


class APITimeoutError(APIClientError):
    """Raised when an API request exceeds the configured timeout."""


class APIStatusError(APIClientError):
    """Raised when the backend API returns an HTTP 4xx or 5xx status code."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        detail: Optional[Any] = None,
    ) -> None:
        super().__init__(f"HTTP {status_code} ({error_code}): {message}")
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.detail = detail


class APIClient:
    """Synchronous HTTP client for interacting with the Fake News Detection FastAPI backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize the API client.

        Args:
            base_url: FastAPI base URL (defaults to API_BASE_URL env var or http://127.0.0.1:8000).
            timeout: Default HTTP request timeout in seconds.
            client: Optional pre-configured httpx.Client (used for testing).
        """
        raw_url = base_url or os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL
        self.base_url = raw_url.rstrip("/")
        self.timeout = timeout
        self._client = client

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Execute HTTP request using injected client or managed httpx client."""
        url = f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"
        if self._client is not None:
            return self._client.request(method, url, **kwargs)
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            return client.request(method, url, **kwargs)

    def _handle_response(self, response: httpx.Response) -> Any:
        """Parse JSON response and raise structured domain errors for failure status codes."""
        if response.is_success:
            try:
                return response.json()
            except Exception as exc:
                raise APIClientError(f"Failed to decode response JSON: {exc}") from exc

        # Parse structured error payload
        error_code = "unknown_error"
        message = response.text or "An unexpected server error occurred."
        detail = None

        try:
            err_data = response.json()
            if isinstance(err_data, dict):
                error_code = err_data.get("error", "unknown_error")
                message = err_data.get("message", message)
                detail = err_data.get("detail")
        except Exception:
            pass

        raise APIStatusError(
            status_code=response.status_code,
            error_code=error_code,
            message=message,
            detail=detail,
        )

    def check_health(self) -> dict[str, Any]:
        """Perform a constant-time liveness check on the backend.

        Returns:
            Dict containing backend health information, or offline status indicator.
        """
        try:
            res = self._request("GET", "/healthz", timeout=5.0)
            if res.is_success:
                data = res.json()
                data["online"] = True
                return data
            return {
                "online": False,
                "status": "error",
                "error": f"HTTP {res.status_code}",
            }
        except httpx.TimeoutException:
            return {
                "online": False,
                "status": "timeout",
                "error": "Liveness check timed out after 5 seconds.",
            }
        except Exception as exc:
            return {
                "online": False,
                "status": "offline",
                "error": f"Cannot connect to backend at {self.base_url}: {exc}",
            }

    def analyze(self, text: str) -> dict[str, Any]:
        """Submit text statement for multi-signal fact-checking and audit persistence.

        Args:
            text: News claim or article text (1 to 20,000 characters).

        Returns:
            AnalysisCreateResponse dictionary containing analysis_id and full AnalysisResponse.

        Raises:
            ValueError: If input text fails client-side validation.
            APIConnectionError: If API backend cannot be reached.
            APITimeoutError: If analysis request times out.
            APIStatusError: If backend responds with an HTTP error.
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}.")

        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(stripped) > MAX_ANALYSIS_TEXT_LENGTH:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {MAX_ANALYSIS_TEXT_LENGTH:,} characters "
                f"(received {len(stripped):,} characters)."
            )

        payload = {"text": stripped}

        try:
            res = self._request(
                "POST",
                "/api/v1/analyze",
                json=payload,
                timeout=self.timeout,
            )
            return self._handle_response(res)
        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Cannot connect to backend API at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"Analysis request timed out after {self.timeout}s.") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIClientError(f"Unexpected network error during analysis: {exc}") from exc

    def list_analyses(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Retrieve paginated analysis history ordered newest-first.

        Args:
            limit: Maximum items to return (1-100).
            offset: Starting pagination offset (>=0).

        Returns:
            AnalysisHistoryResponse dictionary (items, total, limit, offset).
        """
        params = {"limit": limit, "offset": offset}
        try:
            res = self._request(
                "GET",
                "/api/v1/analyses",
                params=params,
                timeout=15.0,
            )
            return self._handle_response(res)
        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Cannot connect to backend API at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError("History request timed out.") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIClientError(f"Unexpected error retrieving history: {exc}") from exc

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Retrieve stored analysis audit record from SQLite without re-running models.

        Args:
            analysis_id: Analysis UUID identifier.

        Returns:
            StoredAnalysisResponse dictionary.
        """
        try:
            res = self._request(
                "GET",
                f"/api/v1/analyses/{analysis_id}",
                timeout=15.0,
            )
            return self._handle_response(res)
        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Cannot connect to backend API at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"Retrieval of analysis {analysis_id} timed out.") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIClientError(f"Unexpected error retrieving analysis: {exc}") from exc

    def delete_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Delete an analysis record and its cascaded feedback records.

        Args:
            analysis_id: Analysis UUID identifier.

        Returns:
            Dict confirming deletion: {"deleted": True, "analysis_id": "..."}.
        """
        try:
            res = self._request(
                "DELETE",
                f"/api/v1/analyses/{analysis_id}",
                timeout=15.0,
            )
            return self._handle_response(res)
        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Cannot connect to backend API at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"Deletion of analysis {analysis_id} timed out.") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIClientError(f"Unexpected error deleting analysis: {exc}") from exc

    def submit_feedback(
        self,
        analysis_id: str,
        rating: str,
        notes: Optional[str] = None,
        corrected_verdict: Optional[str] = None,
    ) -> dict[str, Any]:
        """Submit qualitative feedback on an analysis for post-hoc error analysis.

        Args:
            analysis_id: Target analysis UUID.
            rating: Feedback rating ("helpful", "unhelpful", or "neutral").
            notes: Optional text commentary (up to 2,000 chars).
            corrected_verdict: Optional suggested verdict correction.

        Returns:
            FeedbackResponse dictionary.
        """
        payload: dict[str, Any] = {
            "analysis_id": analysis_id,
            "rating": rating,
        }
        if notes is not None:
            payload["comment"] = notes
        if corrected_verdict is not None and corrected_verdict.strip():
            payload["corrected_verdict"] = corrected_verdict.strip()

        try:
            res = self._request(
                "POST",
                f"/api/v1/analyses/{analysis_id}/feedback",
                json=payload,
                timeout=15.0,
            )
            return self._handle_response(res)
        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Cannot connect to backend API at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError("Feedback submission timed out.") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIClientError(f"Unexpected error submitting feedback: {exc}") from exc

    def get_feedback(self, analysis_id: str) -> list[dict[str, Any]]:
        """Retrieve qualitative feedback records attached to an analysis.

        Args:
            analysis_id: Target analysis UUID.

        Returns:
            List of FeedbackResponse dictionaries.
        """
        try:
            res = self._request(
                "GET",
                f"/api/v1/analyses/{analysis_id}/feedback",
                timeout=15.0,
            )
            return self._handle_response(res)
        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Cannot connect to backend API at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError("Feedback retrieval timed out.") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIClientError(f"Unexpected error retrieving feedback: {exc}") from exc
