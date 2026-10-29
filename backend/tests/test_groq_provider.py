"""Unit tests for the Groq Provider Adapter (Step 17 / V2.1).

Strict isolation guarantee: All Groq SDK operations and responses are completely
mocked. Pytest makes zero external socket or HTTP network connections.
"""

from collections import namedtuple
import logging
import socket
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import pytest

import groq
from pydantic import BaseModel, Field

from backend.app.ai.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.provider_factory import get_provider
from backend.app.ai.providers.groq_provider import (
    DEFAULT_GROQ_MODEL,
    GroqProvider,
    _normalize_schema_for_strict,
)
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.config import LLMConfig
from backend.app.schemas import ConnectivityResponse, LLMCallMetadata


@pytest.fixture(autouse=True)
def guard_against_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: Prevent ANY network socket connection and isolate env during tests."""
    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("External network connection attempted during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    monkeypatch.setenv("LLM_PROVIDER", "mock")


# ---------------------------------------------------------------------------
# Test 1: Provider initializes with valid fake config
# ---------------------------------------------------------------------------
def test_groq_provider_initializes_with_valid_fake_config() -> None:
    mock_client = MagicMock()
    provider = GroqProvider(
        api_key="gsk_test_fake_key_1234567890",
        model_name="openai/gpt-oss-20b",
        client=mock_client,
    )
    assert provider.provider_name == "groq"
    assert provider.model_name == "openai/gpt-oss-20b"
    assert provider._client is mock_client


# ---------------------------------------------------------------------------
# Test 2: Missing API key raises configuration error
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError) as exc_info:
        GroqProvider(api_key="", model_name="openai/gpt-oss-20b", client=MagicMock())
    assert "Groq API key is missing or empty" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError) as exc_info:
        GroqProvider(api_key=None, model_name="openai/gpt-oss-20b", client=MagicMock())
    assert "Groq API key is missing or empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3: Model ID passed correctly
# ---------------------------------------------------------------------------
def test_model_id_passed_correctly() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(
        api_key="gsk_test_key",
        model_name="openai/gpt-oss-20b",
        client=mock_client,
    )
    provider.generate("System prompt", "User query", ConnectivityResponse)

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# Test 4: System prompt passed correctly
# ---------------------------------------------------------------------------
def test_system_prompt_passed_correctly() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    provider.generate(
        system_prompt="Custom system instructions for fact extraction",
        user_prompt="Input claim",
        response_model=ConnectivityResponse,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert any(m["role"] == "system" and m["content"] == "Custom system instructions for fact extraction" for m in messages)


# ---------------------------------------------------------------------------
# Test 5: User content passed correctly
# ---------------------------------------------------------------------------
def test_user_content_passed_correctly() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    provider.generate(
        system_prompt="System prompt",
        user_prompt="The Earth orbits the Sun once every 365.25 days.",
        response_model=ConnectivityResponse,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert any(m["role"] == "user" and m["content"] == "The Earth orbits the Sun once every 365.25 days." for m in messages)


# ---------------------------------------------------------------------------
# Test 6: Structured JSON schema supplied
# ---------------------------------------------------------------------------
def test_structured_json_schema_supplied() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    provider.generate("Sys", "User", ConnectivityResponse)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    rf = call_kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "ConnectivityResponse"
    assert "properties" in rf["json_schema"]["schema"]
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Test 7: Pydantic output validates
# ---------------------------------------------------------------------------
def test_pydantic_output_validates() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    client = LLMClient(provider=provider)

    res = client.generate_structured("Sys", "User", ConnectivityResponse)
    assert isinstance(res, ConnectivityResponse)
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# Test 8: Malformed structured result rejected
# ---------------------------------------------------------------------------
def test_malformed_structured_result_rejected() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Malformed not valid json {invalid"
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured("Sys", "User", ConnectivityResponse)
    assert "not valid JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 9: Authentication error mapped
# ---------------------------------------------------------------------------
def test_authentication_error_mapped() -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=401)
    mock_client.chat.completions.create.side_effect = groq.AuthenticationError(
        "Invalid API Key",
        response=mock_resp,
        body={"error": {"message": "Invalid API Key", "type": "invalid_request_error"}},
    )

    provider = GroqProvider(api_key="gsk_invalid", model_name="openai/gpt-oss-20b", client=mock_client)
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate("Sys", "User", ConnectivityResponse)
    assert "Groq authentication failed" in str(exc_info.value)
    assert "401" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 10: 429/rate-limit error mapped
# ---------------------------------------------------------------------------
def test_rate_limit_error_mapped() -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=429)
    mock_client.chat.completions.create.side_effect = groq.RateLimitError(
        "Rate limit reached for requests per minute",
        response=mock_resp,
        body={"error": {"message": "Rate limit reached"}},
    )

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("Sys", "User", ConnectivityResponse)
    assert "Groq rate limit or quota exhausted" in str(exc_info.value)
    assert "429" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 11: Timeout mapped
# ---------------------------------------------------------------------------
def test_timeout_mapped() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = groq.APITimeoutError(request=MagicMock())

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    with pytest.raises(LLMTimeoutError) as exc_info:
        provider.generate("Sys", "User", ConnectivityResponse)
    assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 12: Server error mapped
# ---------------------------------------------------------------------------
def test_server_error_mapped() -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=500)
    mock_client.chat.completions.create.side_effect = groq.InternalServerError(
        "Internal server error",
        response=mock_resp,
        body={},
    )

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("Sys", "User", ConnectivityResponse)
    assert "Groq internal server error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 13: Invalid model error handled
# ---------------------------------------------------------------------------
def test_invalid_model_error_handled() -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=404)
    mock_client.chat.completions.create.side_effect = groq.NotFoundError(
        "Model 'non-existent-model' not found",
        response=mock_resp,
        body={},
    )

    provider = GroqProvider(api_key="gsk_key", model_name="non-existent-model", client=mock_client)
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate("Sys", "User", ConnectivityResponse)
    assert "not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 14: Usage metadata captured when present
# ---------------------------------------------------------------------------
def test_usage_metadata_captured_when_present() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    usage_obj = MagicMock(prompt_tokens=42, completion_tokens=18, total_tokens=60)
    mock_completion = MagicMock(choices=[mock_choice], usage=usage_obj)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    raw_text, meta = provider.generate("Sys", "User", ConnectivityResponse)

    assert meta.provider == "groq"
    assert meta.model == "openai/gpt-oss-20b"
    assert meta.input_tokens == 42
    assert meta.output_tokens == 18
    assert meta.total_tokens == 60
    assert meta.latency_ms is not None and meta.latency_ms >= 0


# ---------------------------------------------------------------------------
# Test 15: Missing usage metadata tolerated
# ---------------------------------------------------------------------------
def test_missing_usage_metadata_tolerated() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    raw_text, meta = provider.generate("Sys", "User", ConnectivityResponse)

    assert meta.input_tokens is None
    assert meta.output_tokens is None
    assert meta.total_tokens is None
    assert raw_text == '{"status": "ok"}'


# ---------------------------------------------------------------------------
# Test 16: API key never logged
# ---------------------------------------------------------------------------
def test_api_key_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    secret_key = "gsk_SecretSuperKey987654321"
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    with caplog.at_level(logging.DEBUG):
        provider = GroqProvider(api_key=secret_key, model_name="openai/gpt-oss-20b", client=mock_client)
        provider.generate("Sys", "User", ConnectivityResponse)

    for record in caplog.records:
        assert secret_key not in record.message
        assert secret_key not in str(record)


# ---------------------------------------------------------------------------
# Test 17: API key absent from exception messages
# ---------------------------------------------------------------------------
def test_api_key_absent_from_exception_messages() -> None:
    secret_key = "gsk_LeakTargetKey999888777"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError(
        f"Internal transport failure with token {secret_key}"
    )

    provider = GroqProvider(api_key=secret_key, model_name="openai/gpt-oss-20b", client=mock_client)
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate("Sys", "User", ConnectivityResponse)

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


# ---------------------------------------------------------------------------
# Test 18: No chain-of-thought field returned
# ---------------------------------------------------------------------------
def test_no_chain_of_thought_field_returned() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok"}'
    # Suppose raw message object had a reasoning field from model
    setattr(mock_choice.message, "reasoning", "secret internal thinking process")
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    raw_text, meta = provider.generate("Sys", "User", ConnectivityResponse)

    assert raw_text == '{"status": "ok"}'
    # Verify metadata schema has no chain-of-thought attribute
    assert not hasattr(meta, "chain_of_thought")
    assert not hasattr(meta, "reasoning")
    assert "reasoning" not in meta.model_dump()


# ---------------------------------------------------------------------------
# Test 19: Retry policy bounded
# ---------------------------------------------------------------------------
def test_retry_policy_bounded() -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=500)
    mock_client.chat.completions.create.side_effect = groq.InternalServerError(
        "Transient 500 error",
        response=mock_resp,
        body={},
    )

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    cfg = LLMConfig(provider="groq", api_key="gsk_key", model="openai/gpt-oss-20b", max_retries=1)
    client = LLMClient(config=cfg, provider=provider)

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(LLMProviderError):
            client.generate_structured("Sys", "User", ConnectivityResponse)

    # Initial attempt + 1 retry = 2 calls total
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Test 20: No fallback-provider invocation
# ---------------------------------------------------------------------------
def test_no_fallback_provider_invocation() -> None:
    mock_client = MagicMock()
    mock_resp = MagicMock(status_code=429)
    mock_client.chat.completions.create.side_effect = groq.RateLimitError(
        "Rate limit exceeded",
        response=mock_resp,
        body={},
    )

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    cfg = LLMConfig(provider="groq", api_key="gsk_key", model="openai/gpt-oss-20b", max_retries=0)
    client = LLMClient(config=cfg, provider=provider)

    # Calling generate_structured should propagate the error directly without falling back
    with pytest.raises(LLMProviderError) as exc_info:
        client.generate_structured("Sys", "User", ConnectivityResponse)

    assert "Groq rate limit" in str(exc_info.value)
    # Ensure provider is strictly Groq
    assert provider.provider_name == "groq"


# ---------------------------------------------------------------------------
# Test 21: No Gemini invocation
# ---------------------------------------------------------------------------
def test_no_gemini_invocation() -> None:
    config = LLMConfig(provider="gemini", api_key="any-key", model="gemini-1.5-flash")
    with pytest.raises(LLMConfigurationError) as exc_info:
        get_provider(config)
    assert "V2 runtime is configured for Groq-only operation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 22: Deterministic mock result handled
# ---------------------------------------------------------------------------
def test_deterministic_mock_result_handled() -> None:
    config = LLMConfig(provider="mock")
    mock_p = get_provider(config)
    assert isinstance(mock_p, MockLLMProvider)
    assert mock_p.provider_name == "mock"

    client = LLMClient(provider=mock_p)
    res = client.generate_structured("Sys", "User", ConnectivityResponse)
    assert isinstance(res, ConnectivityResponse)
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# Test 23: Optional/null schema fields handled correctly
# ---------------------------------------------------------------------------
class ComplexOptionalModel(BaseModel):
    title: str = Field(description="Required title")
    notes: Optional[str] = Field(default=None, description="Optional notes")


def test_optional_null_schema_fields_handled_correctly() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"title": "Fact Check Title", "notes": null}'
    mock_completion = MagicMock(choices=[mock_choice], usage=None)
    mock_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key="gsk_key", model_name="openai/gpt-oss-20b", client=mock_client)
    client = LLMClient(provider=provider)

    res = client.generate_structured("Sys", "User", ComplexOptionalModel)
    assert isinstance(res, ComplexOptionalModel)
    assert res.title == "Fact Check Title"
    assert res.notes is None

    # Check that schema normalization required both properties per Groq strict mode
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    schema = call_kwargs["response_format"]["json_schema"]["schema"]
    assert "notes" in schema["required"]
    assert "title" in schema["required"]
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Test 24: Strict-schema normalization tested
# ---------------------------------------------------------------------------
def test_strict_schema_normalization_tested() -> None:
    raw_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "sub_obj": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            },
            "items_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_name": {"type": "string"},
                    },
                },
            },
        },
        "$defs": {
            "CustomDef": {
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean"},
                },
            }
        },
    }

    norm = _normalize_schema_for_strict(raw_schema)

    # Root object
    assert norm["additionalProperties"] is False
    assert set(norm["required"]) == {"name", "sub_obj", "items_list"}

    # Nested object
    assert norm["properties"]["sub_obj"]["additionalProperties"] is False
    assert norm["properties"]["sub_obj"]["required"] == ["count"]

    # Array items object
    assert norm["properties"]["items_list"]["items"]["additionalProperties"] is False
    assert norm["properties"]["items_list"]["items"]["required"] == ["item_name"]

    # $defs object
    assert norm["$defs"]["CustomDef"]["additionalProperties"] is False
    assert norm["$defs"]["CustomDef"]["required"] == ["flag"]


# ---------------------------------------------------------------------------
# Test 25: No network during pytest
# ---------------------------------------------------------------------------
def test_no_real_socket_network_connection_during_pytest() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
    assert "External network connection attempted" in str(exc_info.value)
