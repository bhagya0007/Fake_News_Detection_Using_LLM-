"""Unit tests for the Google Gemini Provider Adapter (Step 10.5).

Strict isolation guarantee: All Gemini SDK operations and responses are completely
mocked. Pytest makes zero external socket or HTTP network connections.
"""

from collections import namedtuple
import socket
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from google.genai import errors, types
from pydantic import BaseModel

from backend.app.ai.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.provider_factory import get_provider
from backend.app.ai.providers.gemini_provider import GeminiProvider
from backend.app.config import LLMConfig
from backend.app.schemas import ConnectivityResponse, LLMCallMetadata


@pytest.fixture(autouse=True)
def guard_against_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: Prevent ANY network socket connection and isolate env during tests."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("External network connection attempted during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    # Ensure default env loading doesn't accidentally invoke live Gemini in tests
    monkeypatch.setenv("LLM_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Test 1: GeminiProvider initializes with valid fake config
# ---------------------------------------------------------------------------
def test_gemini_provider_initializes_with_valid_config() -> None:
    mock_client = MagicMock()
    provider = GeminiProvider(
        api_key="fake-key-12345",
        model_name="gemini-3.8-flash",
        client=mock_client,
    )
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-3.8-flash"
    assert provider._client is mock_client


# ---------------------------------------------------------------------------
# Test 2: Missing key rejected
# ---------------------------------------------------------------------------
def test_missing_key_rejected() -> None:
    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiProvider(api_key="", model_name="gemini-3.8-flash", client=MagicMock())
    assert "Gemini API key is missing or empty" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiProvider(api_key=None, model_name="gemini-3.8-flash", client=MagicMock())
    assert "Gemini API key is missing or empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3: Missing model rejected
# ---------------------------------------------------------------------------
def test_missing_model_rejected() -> None:
    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiProvider(api_key="valid-fake-key", model_name="", client=MagicMock())
    assert "Gemini model name is missing or empty" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiProvider(api_key="valid-fake-key", model_name=None, client=MagicMock())
    assert "Gemini model name is missing or empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4: Correct provider name
# ---------------------------------------------------------------------------
def test_correct_provider_name() -> None:
    provider = GeminiProvider(api_key="key", model_name="model", client=MagicMock())
    assert provider.provider_name == "gemini"


# ---------------------------------------------------------------------------
# Test 5: Configured model preserved
# ---------------------------------------------------------------------------
def test_configured_model_preserved() -> None:
    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=MagicMock())
    assert provider.model_name == "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# Test 6: SDK client receives configured API key
# ---------------------------------------------------------------------------
def test_sdk_client_receives_configured_api_key() -> None:
    with patch("backend.app.ai.providers.gemini_provider.genai.Client") as mock_client_cls:
        provider = GeminiProvider(api_key="my-specific-test-key", model_name="gemini-3.8-flash")
        mock_client_cls.assert_called_once()
        assert mock_client_cls.call_args.kwargs["api_key"] == "my-specific-test-key"
        assert provider.model_name == "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# Test 7: System prompt forwarded
# ---------------------------------------------------------------------------
def test_system_prompt_forwarded() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="fake-key", model_name="gemini-3.8-flash", client=mock_client)
    provider.generate(system_prompt="My system instruction", user_prompt="My query", response_model=ConnectivityResponse)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == "My system instruction"


# ---------------------------------------------------------------------------
# Test 8: User prompt forwarded
# ---------------------------------------------------------------------------
def test_user_prompt_forwarded() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="fake-key", model_name="gemini-3.8-flash", client=mock_client)
    provider.generate(system_prompt="Instruction", user_prompt="User query prompt text", response_model=ConnectivityResponse)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["contents"] == "User query prompt text"


# ---------------------------------------------------------------------------
# Test 9: Response schema forwarded
# ---------------------------------------------------------------------------
def test_response_schema_forwarded() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="fake-key", model_name="gemini-3.8-flash", client=mock_client)
    provider.generate(system_prompt="Instruction", user_prompt="Query", response_model=ConnectivityResponse)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert isinstance(config.response_schema, types.Schema)
    assert config.response_schema.additional_properties is None
    assert "status" in config.response_schema.properties


# ---------------------------------------------------------------------------
# Test 10: Valid structured output accepted
# ---------------------------------------------------------------------------
def test_valid_structured_output_accepted() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    client = LLMClient(provider=provider)

    res = client.generate_structured("sys", "usr", ConnectivityResponse)
    assert isinstance(res, ConnectivityResponse)
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# Test 11: Empty output rejected
# ---------------------------------------------------------------------------
def test_empty_output_rejected() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "   "
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    with pytest.raises(LLMResponseValidationError) as exc_info:
        provider.generate("sys", "usr", ConnectivityResponse)
    assert "returned an empty response" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 12: Malformed output rejected
# ---------------------------------------------------------------------------
def test_malformed_output_rejected() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "this is not json"
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured("sys", "usr", ConnectivityResponse)
    assert "not valid JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 13: SDK auth error mapped safely
# ---------------------------------------------------------------------------
def test_sdk_auth_error_mapped_safely() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.ClientError(
        401,
        {"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}},
    )

    provider = GeminiProvider(api_key="secret-key", model_name="gemini-3.8-flash", client=mock_client)
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate("sys", "usr", ConnectivityResponse)
    assert "Gemini authentication failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 14: Timeout error mapped safely
# ---------------------------------------------------------------------------
def test_timeout_error_mapped_safely() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Network connection timed out")

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    with pytest.raises(LLMTimeoutError) as exc_info:
        provider.generate("sys", "usr", ConnectivityResponse)
    assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 15: Provider error mapped safely
# ---------------------------------------------------------------------------
def test_provider_error_mapped_safely() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "Service Unavailable", "status": "UNAVAILABLE"}},
    )

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("sys", "usr", ConnectivityResponse)
    assert "Gemini provider error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 16: API key absent from exception string
# ---------------------------------------------------------------------------
def test_api_key_absent_from_exception_string() -> None:
    secret_key = "AIzaSySecretProductionKey12345"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        f"Unauthorized call with token {secret_key}"
    )

    provider = GeminiProvider(api_key=secret_key, model_name="gemini-3.8-flash", client=mock_client)
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("sys", "usr", ConnectivityResponse)

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


# ---------------------------------------------------------------------------
# Test 17: Metadata contains provider/model
# ---------------------------------------------------------------------------
def test_metadata_contains_provider_model() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    raw_text, meta = provider.generate("sys", "usr", ConnectivityResponse)

    assert isinstance(meta, LLMCallMetadata)
    assert meta.provider == "gemini"
    assert meta.model == "gemini-3.8-flash"
    assert meta.latency_ms is not None and meta.latency_ms >= 0


# ---------------------------------------------------------------------------
# Test 18: Mock token metadata handled
# ---------------------------------------------------------------------------
def test_mock_token_metadata_handled() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'

    UsageTuple = namedtuple("UsageMetadata", ["prompt_token_count", "candidates_token_count", "total_token_count"])
    mock_response.usage_metadata = UsageTuple(prompt_token_count=42, candidates_token_count=18, total_token_count=60)
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(api_key="key", model_name="gemini-3.8-flash", client=mock_client)
    raw_text, meta = provider.generate("sys", "usr", ConnectivityResponse)

    assert meta.input_tokens == 42
    assert meta.output_tokens == 18
    assert meta.total_tokens == 60


# ---------------------------------------------------------------------------
# Test 19: No real socket/network connection during pytest
# ---------------------------------------------------------------------------
def test_no_real_socket_network_connection_during_pytest() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
    assert "External network connection attempted" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 20: Provider factory selects GeminiProvider when LLM_PROVIDER=gemini
# ---------------------------------------------------------------------------
def test_provider_factory_selects_gemini_provider() -> None:
    config = LLMConfig(
        provider="gemini",
        api_key="fake-test-key",
        model="gemini-3.8-flash",
    )
    provider = get_provider(config)
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-3.8-flash"
