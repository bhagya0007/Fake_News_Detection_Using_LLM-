"""Unit tests for GDELTClient (V2.2).

Verifies HTTP request construction, payload parsing, unknown/missing field tolerance,
error and timeout mapping, in-memory caching, rate limit shielding, and zero network calls.
"""

from __future__ import annotations

import socket
import time
from typing import Any
from unittest.mock import MagicMock
import pytest

import httpx

from backend.app.services.gdelt_client import GDELTClient


@pytest.fixture(autouse=True)
def guard_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: prohibit any network connections during unit tests."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Prohibited real network call during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


def test_context_request_parameters() -> None:
    """Requirement 6: Verify Context API request parameters match official contract."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://reuters.com/business/article-1",
                "title": "Pandora opens factory",
                "domain": "reuters.com",
                "sentence": "Pandora opened its new facility today.",
                "context": "According to officials, Pandora opened its new facility today in Vietnam.",
                "seendate": "20261018T070000Z",
            }
        ]
    }
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    results = client.search_context("Pandora factory", timespan="72h", max_records=10)

    assert len(results) == 1
    mock_http.get.assert_called_once()
    called_args, called_kwargs = mock_http.get.call_args
    assert called_kwargs["params"]["query"] == "Pandora factory"
    assert called_kwargs["params"]["format"] == "json"
    assert called_kwargs["params"]["maxrecords"] == 10
    assert called_kwargs["params"]["timespan"] == "72h"


def test_context_json_parsed() -> None:
    """Requirement 8: Verify Context 2.0 response fields are normalized correctly."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://bbc.com/news/world-123",
                "title": "Renewable energy milestone reached",
                "domain": "bbc.com",
                "sentence": "Wind and solar accounted for 50 percent of electricity.",
                "context": "In a historic shift, wind and solar accounted for 50 percent of electricity this quarter.",
                "seendate": "20261015T120000Z",
            }
        ]
    }
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Renewable energy")

    assert len(items) == 1
    item = items[0]
    assert item["url"] == "https://bbc.com/news/world-123"
    assert item["title"] == "Renewable energy milestone reached"
    assert item["domain"] == "bbc.com"
    assert "In a historic shift" in item["snippet"]
    assert item["published_date"] == "2026-10-15"
    assert item["retrieval_source"] == "gdelt_context"
    assert item["evidence_quality"] == "context_snippet"


def test_doc_request_parameters_and_parsing() -> None:
    """Requirement 7 & 9: Verify DOC 2.0 parameters and headline-only normalization."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://apnews.com/article/ai-summit-456",
                "title": "Global leaders sign artificial intelligence safety pact",
                "domain": "apnews.com",
                "seendate": "20261017T083000Z",
                "language": "English",
            }
        ]
    }
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_doc("AI safety summit", timespan="24h", max_records=5)

    assert len(items) == 1
    item = items[0]
    assert item["url"] == "https://apnews.com/article/ai-summit-456"
    assert item["title"] == "Global leaders sign artificial intelligence safety pact"
    assert item["domain"] == "apnews.com"
    assert item["snippet"] == item["title"]  # Headline-only fallback
    assert item["published_date"] == "2026-10-17"
    assert item["retrieval_source"] == "gdelt_doc"
    assert item["evidence_quality"] == "headline_only"

    called_kwargs = mock_http.get.call_args.kwargs
    assert called_kwargs["params"]["mode"] == "artlist"
    assert called_kwargs["params"]["format"] == "json"


def test_unknown_fields_tolerated() -> None:
    """Requirement 10: Unexpected or future GDELT fields do not disrupt parsing."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://example.com/item",
                "title": "Test Title",
                "domain": "example.com",
                "sentence": "Sample snippet",
                "unknown_extra_metric": 99.8,
                "future_gdelt_v3_flag": True,
            }
        ]
    }
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Test query")
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/item"


def test_missing_fields_tolerated() -> None:
    """Requirement 11: Missing snippet, seendate, or domain fall back safely."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://clean-domain.org/story",
                # title, domain, seendate, context omitted
            }
        ]
    }
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Incomplete article")
    assert len(items) == 1
    item = items[0]
    assert item["url"] == "https://clean-domain.org/story"
    assert item["domain"] == "clean-domain.org"  # Derived safely from URL
    assert item["title"] == "Untitled Article"
    assert item["published_date"] is None


def test_timeout_handling() -> None:
    """Requirement 12: Network timeouts return empty list safely without raising exceptions."""
    mock_http = MagicMock()
    mock_http.get.side_effect = httpx.TimeoutException("Connection timed out")

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Timeout query")
    assert items == []


def test_429_rate_limit_handling() -> None:
    """Requirement 13: HTTP 429 rate limit returns empty list gracefully."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Rate limited query")
    assert items == []


def test_5xx_server_error_and_retry() -> None:
    """Requirement 14: HTTP 500 error triggers bounded retry and returns empty if persistent."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Server error query")
    assert items == []
    assert mock_http.get.call_count == 2  # Retried once, then stopped


def test_invalid_json_handling() -> None:
    """Requirement 15: Malformed JSON payload returns empty list without crashing."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON string")
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Bad JSON query")
    assert items == []


def test_zero_results_handled_as_empty_list() -> None:
    """Requirement 16: Zero matches from GDELT return empty list without error."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"articles": []}
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Nonexistent topic xyz123")
    assert items == []


def test_in_memory_ttl_cache() -> None:
    """Requirement 24: In-memory TTL cache avoids redundant outbound HTTP requests."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "articles": [
            {
                "url": "https://cached-news.com/item",
                "title": "Cached News Title",
                "domain": "cached-news.com",
            }
        ]
    }
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, cache_ttl_seconds=60, min_request_interval=0.0)
    res1 = client.search_context("Cache test query")
    assert len(res1) == 1
    assert mock_http.get.call_count == 1

    # Second call with identical query hits cache
    res2 = client.search_context("Cache test query")
    assert len(res2) == 1
    assert mock_http.get.call_count == 1  # No additional HTTP request made


def test_zero_api_key_sent_or_required() -> None:
    """Requirement 35: GDELT client requires no API key and never transmits secrets."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"articles": []}
    mock_http.get.return_value = mock_resp
    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    client.search_context("Clean query")

    call_kwargs = mock_http.get.call_args.kwargs
    params = call_kwargs["params"]
    assert "api_key" not in params
    assert "key" not in params
    assert "token" not in params


def test_429_sets_rate_limited_status() -> None:
    """HTTP 429 sets structured status and error message."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Test rate limit")
    assert items == []
    assert client.last_status == "rate_limited"
    assert "429" in (client.last_error or "")


def test_timeout_sets_timeout_status() -> None:
    """Timeout exception sets structured status and message."""
    mock_http = MagicMock()
    mock_http.get.side_effect = httpx.TimeoutException("Read timed out")

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_doc("Test timeout")
    assert items == []
    assert client.last_status == "timeout"
    assert "timed out" in (client.last_error or "").lower()


def test_doc_timespan_configurable_and_used() -> None:
    """DOC search defaults to configured doc_timespan (e.g. 3m)."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"articles": []}
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, doc_timespan="3m", min_request_interval=0.0)
    client.search_doc("Test query")
    call_kwargs = mock_http.get.call_args.kwargs
    assert call_kwargs["params"]["timespan"] == "3m"


def test_invalid_json_sets_status() -> None:
    """Non-JSON response sets invalid_json status."""
    mock_http = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Not JSON")
    mock_http.get.return_value = mock_resp

    client = GDELTClient(http_client=mock_http, min_request_interval=0.0)
    items = client.search_context("Test bad json")
    assert items == []
    assert client.last_status == "invalid_json"

