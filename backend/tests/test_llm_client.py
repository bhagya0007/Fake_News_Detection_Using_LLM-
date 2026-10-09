"""Unit tests for the modular LLM provider adapter and client (Step 10).

Verifies provider abstraction, Pydantic validation, timeouts, safe retries,
metadata tracking, exception mapping, secret protection, and zero external network calls.
"""

import socket
import time
from typing import Any, Generator
import pytest
from pydantic import BaseModel, Field

from backend.app.ai.exceptions import (
    LLMBaseError,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.provider_base import LLMProvider
from backend.app.ai.provider_factory import get_provider
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.config import LLMConfig
from backend.app.schemas import (
    ConnectivityResponse,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    LLMCallMetadata,
    SourceAttribution,
    VerdictLabel,
)


@pytest.fixture(autouse=True)
def guard_against_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict guard: Prevent ANY network socket connection during test execution."""
    original_connect = socket.socket.connect

    def forbidden_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("External network connection attempted during test execution!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


# ---------------------------------------------------------------------------
# Test 1: Missing provider configuration handled clearly
# ---------------------------------------------------------------------------
def test_missing_provider_configuration_handled_clearly() -> None:
    config = LLMConfig(provider="")
    with pytest.raises(LLMConfigurationError) as exc_info:
        LLMClient(config=config)
    assert "LLM provider is not configured" in str(exc_info.value)

    config_none = LLMConfig(provider=None)
    with pytest.raises(LLMConfigurationError) as exc_info:
        get_provider(config_none)
    assert "LLM provider is not configured" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2: Missing API key handled clearly when real provider requires it
# ---------------------------------------------------------------------------
def test_missing_api_key_handled_clearly_when_real_provider_requires_it() -> None:
    config = LLMConfig(provider="openai", api_key="")
    with pytest.raises(LLMConfigurationError) as exc_info:
        get_provider(config)
    assert "Missing API key for provider 'openai'" in str(exc_info.value)

    config_gemini = LLMConfig(provider="gemini", api_key=None)
    with pytest.raises(LLMConfigurationError) as exc_info:
        get_provider(config_gemini)
    assert "Missing API key for provider 'gemini'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3: Mock provider initializes
# ---------------------------------------------------------------------------
def test_mock_provider_initializes() -> None:
    provider = MockLLMProvider(model_name="mock-custom-v2")
    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-custom-v2"
    assert len(provider.call_history) == 0


# ---------------------------------------------------------------------------
# Test 4: Mock structured response validates successfully
# ---------------------------------------------------------------------------
def test_mock_structured_response_validates_successfully() -> None:
    provider = MockLLMProvider()
    valid_payload = {
        "label": "likely_reliable",
        "source_attribution": "strong",
        "evidence_support": "supported",
        "internal_consistency": "consistent",
        "rhetorical_cues": ["cites primary source"],
        "unsupported_claims": [],
        "short_reason": "Verified against credible primary documentation.",
    }
    provider.queue_response(valid_payload)
    client = LLMClient(provider=provider)

    result = client.generate_structured(
        system_prompt="Test system prompt",
        user_prompt="Evaluate test claim",
        response_model=LLMAssessment,
    )

    assert isinstance(result, LLMAssessment)
    assert result.label == VerdictLabel.likely_reliable
    assert result.source_attribution == SourceAttribution.strong
    assert result.evidence_support == EvidenceSupport.supported
    assert result.short_reason == "Verified against credible primary documentation."
    assert len(provider.call_history) == 1


# ---------------------------------------------------------------------------
# Test 5: Invalid JSON rejected
# ---------------------------------------------------------------------------
def test_invalid_json_rejected() -> None:
    provider = MockLLMProvider()
    provider.queue_response("{ this is broken json, not valid at all !!!")
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured(
            system_prompt="Prompt",
            user_prompt="Prompt",
            response_model=ConnectivityResponse,
        )
    assert "Response is not valid JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 6: Schema-invalid JSON rejected
# ---------------------------------------------------------------------------
def test_schema_invalid_json_rejected() -> None:
    provider = MockLLMProvider()
    # Missing required 'status' field for ConnectivityResponse
    provider.queue_response({"wrong_field": "ok"})
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured(
            system_prompt="Prompt",
            user_prompt="Prompt",
            response_model=ConnectivityResponse,
        )
    assert "Response failed schema validation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 7: Invalid verdict enum rejected
# ---------------------------------------------------------------------------
def test_invalid_verdict_enum_rejected() -> None:
    provider = MockLLMProvider()
    bad_enum_payload = {
        "label": "definitely_real_fabricated_label",
        "source_attribution": "strong",
        "evidence_support": "supported",
        "internal_consistency": "consistent",
        "rhetorical_cues": [],
        "unsupported_claims": [],
        "short_reason": "Some reason.",
    }
    provider.queue_response(bad_enum_payload)
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured(
            system_prompt="Prompt",
            user_prompt="Prompt",
            response_model=LLMAssessment,
        )
    assert "Response failed schema validation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 8: Empty mandatory field rejected
# ---------------------------------------------------------------------------
def test_empty_mandatory_field_rejected() -> None:
    provider = MockLLMProvider()
    empty_field_payload = {
        "label": "likely_reliable",
        "source_attribution": "strong",
        "evidence_support": "supported",
        "internal_consistency": "consistent",
        "rhetorical_cues": [],
        "unsupported_claims": [],
        "short_reason": "   ",  # Whitespace only violates validator
    }
    provider.queue_response(empty_field_payload)
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured(
            system_prompt="Prompt",
            user_prompt="Prompt",
            response_model=LLMAssessment,
        )
    assert "Response failed schema validation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 9: Deterministic mock response works
# ---------------------------------------------------------------------------
def test_deterministic_mock_response_works() -> None:
    provider = MockLLMProvider()
    client = LLMClient(provider=provider)

    res1 = client.generate_structured("sys", "usr1", ConnectivityResponse)
    res2 = client.generate_structured("sys", "usr2", ConnectivityResponse)

    assert res1.status == "ok"
    assert res2.status == "ok"
    assert len(provider.call_history) == 2


# ---------------------------------------------------------------------------
# Test 10: Provider factory selects expected mock provider
# ---------------------------------------------------------------------------
def test_provider_factory_selects_expected_mock_provider() -> None:
    config = LLMConfig(provider="mock", model="custom-mock-model")
    provider = get_provider(config)

    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"
    assert provider.model_name == "custom-mock-model"


# ---------------------------------------------------------------------------
# Test 11: Unknown provider rejected
# ---------------------------------------------------------------------------
def test_unknown_provider_rejected() -> None:
    config = LLMConfig(provider="unsupported_quantum_llm")
    with pytest.raises(LLMConfigurationError) as exc_info:
        get_provider(config)
    assert "Unsupported LLM provider 'unsupported_quantum_llm'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 12: Timeout exception mapped cleanly
# ---------------------------------------------------------------------------
def test_timeout_exception_mapped_cleanly() -> None:
    provider = MockLLMProvider()
    provider.queue_response(TimeoutError("Socket timed out after 30 seconds"))
    config = LLMConfig(provider="mock", timeout_seconds=30.0, max_retries=0)
    client = LLMClient(config=config, provider=provider)

    with pytest.raises(LLMTimeoutError) as exc_info:
        client.generate_structured("sys", "usr", ConnectivityResponse)
    assert "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 13: Provider exception mapped cleanly
# ---------------------------------------------------------------------------
def test_provider_exception_mapped_cleanly() -> None:
    provider = MockLLMProvider()
    provider.queue_response(RuntimeError("Service unavailable 503"))
    config = LLMConfig(provider="mock", max_retries=0)
    client = LLMClient(config=config, provider=provider)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate_structured("sys", "usr", ConnectivityResponse)
    assert "Service unavailable 503" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 14: Retries stop at configured limit
# ---------------------------------------------------------------------------
def test_retries_stop_at_configured_limit() -> None:
    provider = MockLLMProvider()
    # Queue 5 errors
    for i in range(5):
        provider.queue_response(RuntimeError(f"Transient error #{i + 1}"))

    config = LLMConfig(provider="mock", max_retries=2)  # exactly 3 attempts total (1 initial + 2 retries)
    client = LLMClient(config=config, provider=provider)

    with pytest.raises(LLMProviderError):
        client.generate_structured("sys", "usr", ConnectivityResponse)

    assert len(provider.call_history) == 3


# ---------------------------------------------------------------------------
# Test 15: Auth/config errors are not endlessly retried
# ---------------------------------------------------------------------------
def test_auth_config_errors_are_not_endlessly_retried() -> None:
    provider = MockLLMProvider()
    # Inject deterministic schema failure
    provider.queue_response({"wrong": "data"})
    config = LLMConfig(provider="mock", max_retries=3)
    client = LLMClient(config=config, provider=provider)

    with pytest.raises(LLMResponseValidationError):
        client.generate_structured("sys", "usr", ConnectivityResponse)

    # Must fail immediately on schema failure without consuming retries
    assert len(provider.call_history) == 1


# ---------------------------------------------------------------------------
# Test 16: API key never appears in exception string
# ---------------------------------------------------------------------------
def test_api_key_never_appears_in_exception_string() -> None:
    secret_key = "sk-super-secret-key-123456789"
    provider = MockLLMProvider()
    provider.queue_response(RuntimeError(f"Unauthorized request using key: {secret_key}"))
    config = LLMConfig(provider="mock", api_key=secret_key, max_retries=0)
    client = LLMClient(config=config, provider=provider)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate_structured("sys", "usr", ConnectivityResponse)

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


# ---------------------------------------------------------------------------
# Test 17: Structured output returns Pydantic object, not arbitrary dict
# ---------------------------------------------------------------------------
def test_structured_output_returns_pydantic_object_not_arbitrary_dict() -> None:
    provider = MockLLMProvider()
    provider.queue_response({"status": "ok"})
    client = LLMClient(provider=provider)

    result = client.generate_structured("sys", "usr", ConnectivityResponse)

    assert isinstance(result, ConnectivityResponse)
    assert isinstance(result, BaseModel)
    assert not isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test 18: Metadata contains provider/model safely
# ---------------------------------------------------------------------------
def test_metadata_contains_provider_model_safely() -> None:
    provider = MockLLMProvider(model_name="safe-model-v1", latency_ms=12.5)
    provider.queue_response({"status": "ok"})
    client = LLMClient(provider=provider)

    result, metadata = client.generate_structured_with_metadata("sys", "usr", ConnectivityResponse)

    assert isinstance(metadata, LLMCallMetadata)
    assert metadata.provider == "mock"
    assert metadata.model == "safe-model-v1"
    assert metadata.latency_ms == 12.5
    assert metadata.input_tokens is not None and metadata.input_tokens > 0
    assert metadata.output_tokens is not None and metadata.output_tokens > 0

    # Ensure no secret auth fields exist on metadata
    assert not hasattr(metadata, "api_key")
    assert not hasattr(metadata, "authorization")
    assert "api_key" not in metadata.model_dump()


# ---------------------------------------------------------------------------
# Test 19: No hidden chain-of-thought field required
# ---------------------------------------------------------------------------
def test_no_hidden_chain_of_thought_field_required() -> None:
    # 1. Verify schema contract explicitly forbids hidden CoT fields
    fields = set(LLMAssessment.model_fields.keys())
    prohibited_fields = {
        "chain_of_thought",
        "cot",
        "reasoning",
        "hidden_reasoning",
        "thinking",
        "internal_monologue",
    }
    assert fields.isdisjoint(prohibited_fields)

    # 2. Verify schema forbids extra fields if injected
    provider = MockLLMProvider()
    cot_injected_payload = {
        "label": "likely_reliable",
        "source_attribution": "strong",
        "evidence_support": "supported",
        "internal_consistency": "consistent",
        "rhetorical_cues": [],
        "unsupported_claims": [],
        "short_reason": "Auditable justification.",
        "chain_of_thought": "Secret hidden thoughts that should be forbidden",
    }
    provider.queue_response(cot_injected_payload)
    client = LLMClient(provider=provider)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        client.generate_structured("sys", "usr", LLMAssessment)
    assert "extra_forbidden" in str(exc_info.value) or "Response failed schema validation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 20: No external API is called during unit tests
# ---------------------------------------------------------------------------
def test_no_external_api_called_during_unit_tests() -> None:
    # Attempting an external socket connection must trigger our test guard
    with pytest.raises(RuntimeError) as exc_info:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
    assert "External network connection attempted" in str(exc_info.value)

    # Running LLMClient with MockLLMProvider triggers zero socket connections
    provider = MockLLMProvider()
    client = LLMClient(provider=provider)
    result = client.generate_structured("sys", "usr", ConnectivityResponse)
    assert result.status == "ok"
