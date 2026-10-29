"""Isolated HTTP Client for GDELT Context 2.0 and DOC 2.0 Discovery (V2.2).

Performs lightweight, rate-conscious queries against official public GDELT endpoints.
Normalizes candidate articles into consistent provenance-bearing records with zero
credential dependencies (GDELT requires NO API key). Features in-memory TTL caching,
polite pacing, and conservative retry handling for transient failures.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from backend.app.config import (
    GDELT_CACHE_TTL_SECONDS,
    GDELT_CONTEXT_TIMESPAN,
    GDELT_CONTEXT_URL,
    GDELT_DOC_FALLBACK,
    GDELT_DOC_TIMESPAN,
    GDELT_DOC_URL,
    GDELT_MAX_RECORDS,
    GDELT_MIN_REQUEST_INTERVAL,
    GDELT_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

USER_AGENT = "FakeNewsDetectionAcademicResearch/1.0 (P_105; Academic Prototype)"
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = GDELT_MIN_REQUEST_INTERVAL  # Polite pacing


def _extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract clean, lowercased network location domain from a URL."""
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc if netloc else None
    except Exception:
        return None


def _format_seendate(raw_date: Optional[str]) -> Optional[str]:
    """Parse GDELT seendate format (e.g. '20261018T071647Z' or '20261018') to ISO date."""
    if not raw_date or not isinstance(raw_date, str):
        return None
    cleaned = raw_date.strip()
    if len(cleaned) >= 8 and cleaned[:8].isdigit():
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return cleaned


class GDELTClient:
    """Client for discovering candidate news articles via GDELT Context 2.0 and DOC 2.0."""

    def __init__(
        self,
        context_url: str = GDELT_CONTEXT_URL,
        doc_url: str = GDELT_DOC_URL,
        timeout_seconds: float = GDELT_TIMEOUT_SECONDS,
        timeout: Optional[float] = None,
        max_records: int = GDELT_MAX_RECORDS,
        context_timespan: str = GDELT_CONTEXT_TIMESPAN,
        doc_timespan: str = GDELT_DOC_TIMESPAN,
        doc_fallback: bool = GDELT_DOC_FALLBACK,
        cache_ttl_seconds: int = GDELT_CACHE_TTL_SECONDS,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.context_url = context_url
        self.doc_url = doc_url
        effective_timeout = timeout if timeout is not None else timeout_seconds
        self.timeout_seconds = float(effective_timeout)
        self.max_records = int(max_records)
        self.context_timespan = context_timespan
        self.doc_timespan = doc_timespan
        self.doc_fallback = doc_fallback
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.min_request_interval = float(min_request_interval)

        self._client = http_client if http_client is not None else httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        self._last_request_time: float = 0.0
        # In-memory TTL cache: cache_key -> (timestamp, list_of_records)
        self._cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

        # Last request telemetry
        self.last_status: str = "init"
        self.last_error: Optional[str] = None
        self.last_elapsed_ms: float = 0.0

    def search_context(
        self,
        query: str,
        timespan: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query GDELT Context 2.0 API for sentence-level context matches.

        Args:
            query: Sanitized search query.
            timespan: Optional timespan string (e.g. '72h', '24h').
            max_records: Maximum number of records to request.

        Returns:
            List of normalized candidate evidence dictionaries.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        span = timespan or self.context_timespan
        limit = max_records or self.max_records

        cache_key = f"context:{clean_query}:{span}:{limit}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self.last_status = "success"
            self.last_error = None
            self.last_elapsed_ms = 0.0
            return cached

        params = {
            "query": clean_query,
            "format": "json",
            "maxrecords": limit,
        }
        if span:
            params["timespan"] = span

        data = self._execute_request(self.context_url, params)
        records = self._parse_context_response(data)
        self._set_cache(cache_key, records)
        return records

    def search_doc(
        self,
        query: str,
        timespan: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query GDELT DOC 2.0 API for article-level headline/metadata matches.

        Args:
            query: Sanitized search query.
            timespan: Optional timespan string (e.g. '72h', '3m', '7d').
            max_records: Maximum number of records to request.

        Returns:
            List of normalized headline-only candidate evidence dictionaries.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        span = timespan or self.doc_timespan or self.context_timespan
        limit = max_records or self.max_records

        cache_key = f"doc:{clean_query}:{span}:{limit}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self.last_status = "success"
            self.last_error = None
            self.last_elapsed_ms = 0.0
            return cached

        params = {
            "query": clean_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": limit,
        }
        if span:
            params["timespan"] = span

        data = self._execute_request(self.doc_url, params)
        records = self._parse_doc_response(data)
        self._set_cache(cache_key, records)
        return records

    def _execute_request(self, endpoint_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute HTTP GET with pacing, transient error handling, and safe retry."""
        self._pace_request()
        self.last_status = "pending"
        self.last_error = None

        t0 = time.perf_counter()
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = self._client.get(endpoint_url, params=params)
                self.last_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                if response.status_code == 200:
                    try:
                        self.last_status = "success"
                        return response.json()
                    except Exception as json_err:
                        self.last_status = "invalid_json"
                        self.last_error = f"Invalid JSON: {json_err}"
                        logger.warning("GDELT response is not valid JSON from %s: %s", endpoint_url, json_err)
                        return None
                elif response.status_code == 429:
                    self.last_status = "rate_limited"
                    self.last_error = "GDELT rate limit encountered (HTTP 429)"
                    logger.warning("GDELT rate limit encountered (HTTP 429) on %s", endpoint_url)
                    return None
                elif response.status_code >= 500:
                    if attempt < max_attempts - 1:
                        time.sleep(1.0)
                        continue
                    self.last_status = "server_error"
                    self.last_error = f"GDELT server error (HTTP {response.status_code})"
                    logger.warning("GDELT server error (HTTP %d) on %s", response.status_code, endpoint_url)
                    return None
                else:
                    self.last_status = f"http_{response.status_code}"
                    self.last_error = f"GDELT unexpected status (HTTP {response.status_code})"
                    logger.warning("GDELT unexpected status (HTTP %d) on %s", response.status_code, endpoint_url)
                    return None
            except httpx.TimeoutException:
                self.last_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
                    continue
                self.last_status = "timeout"
                self.last_error = f"GDELT request timed out on {endpoint_url} after {self.timeout_seconds:.1f}s"
                logger.warning("GDELT request timed out on %s after %.1fs", endpoint_url, self.timeout_seconds)
                return None
            except Exception as exc:
                self.last_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                self.last_status = "network_error"
                self.last_error = str(exc)
                logger.warning("GDELT network error on %s: %s", endpoint_url, exc)
                return None
        return None

    def _parse_context_response(self, data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize raw Context 2.0 JSON payload into standardized candidate items."""
        if not data or not isinstance(data, dict):
            return []

        articles = data.get("articles") or data.get("results") or []
        if not isinstance(articles, list):
            return []

        results: List[Dict[str, Any]] = []
        for raw in articles:
            if not isinstance(raw, dict):
                continue

            raw_url = raw.get("url")
            if not raw_url or not isinstance(raw_url, str):
                continue

            cleaned_url = raw_url.strip()
            if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
                continue

            title = str(raw.get("title", "")).strip() or "Untitled Article"
            domain = str(raw.get("domain", "")).strip() or _extract_domain(cleaned_url)

            # Prefer context snippet, fallback to sentence, then title
            context_text = str(raw.get("context", "")).strip()
            sentence_text = str(raw.get("sentence", "")).strip()
            snippet = context_text if context_text else (sentence_text if sentence_text else title)

            results.append({
                "url": cleaned_url,
                "title": title,
                "domain": domain,
                "snippet": snippet,
                "published_date": _format_seendate(raw.get("seendate")),
                "language": raw.get("language"),
                "source_type": "news",
                "evidence_quality": "context_snippet",
                "retrieval_source": "gdelt_context",
            })
        return results

    def _parse_doc_response(self, data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize raw DOC 2.0 JSON payload into headline-only candidate items."""
        if not data or not isinstance(data, dict):
            return []

        articles = data.get("articles") or data.get("results") or []
        if not isinstance(articles, list):
            return []

        results: List[Dict[str, Any]] = []
        for raw in articles:
            if not isinstance(raw, dict):
                continue

            raw_url = raw.get("url")
            if not raw_url or not isinstance(raw_url, str):
                continue

            cleaned_url = raw_url.strip()
            if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
                continue

            title = str(raw.get("title", "")).strip() or "Untitled Article"
            domain = str(raw.get("domain", "")).strip() or _extract_domain(cleaned_url)

            results.append({
                "url": cleaned_url,
                "title": title,
                "domain": domain,
                "snippet": title,  # DOC API provides headline only
                "published_date": _format_seendate(raw.get("seendate")),
                "language": raw.get("language"),
                "source_type": "news",
                "evidence_quality": "headline_only",
                "retrieval_source": "gdelt_doc",
            })
        return results

    def _pace_request(self) -> None:
        """Enforce polite minimum delay between outbound requests."""
        now = time.perf_counter()
        elapsed = now - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.perf_counter()

    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve valid unexpired records from in-memory cache."""
        if key not in self._cache:
            return None
        cached_time, records = self._cache[key]
        if time.time() - cached_time > self.cache_ttl_seconds:
            del self._cache[key]
            return None
        return records

    def _set_cache(self, key: str, records: List[Dict[str, Any]]) -> None:
        """Store query results in cache with current timestamp."""
        self._cache[key] = (time.time(), records)
