"""Deterministic Mock LLM Provider for offline testing, schema validation, and development.

This provider never makes external network calls and does not consume API credits.
"""

from collections import deque
import json
import time
from typing import Any, Optional, Type, Union

from pydantic import BaseModel

from backend.app.ai.exceptions import LLMProviderError
from backend.app.ai.provider_base import BaseLLMProvider
from backend.app.schemas import (
    Claim,
    ClaimExtractionResult,
    ConnectivityResponse,
    EvidenceSupport,
    InternalConsistency,
    LLMAssessment,
    LLMCallMetadata,
    SourceAttribution,
    VerdictLabel,
)


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, offline mock provider for testing and validation."""

    def __init__(
        self,
        model_name: str = "mock-model-v1",
        default_response: Optional[Union[str, dict[str, Any], BaseModel, Exception]] = None,
        latency_ms: float = 5.0,
    ) -> None:
        self._model_name = model_name
        self._default_response = default_response
        self._latency_ms = latency_ms
        self._response_queue: deque[Union[str, dict[str, Any], BaseModel, Exception]] = deque()
        self.call_history: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def queue_response(self, response: Union[str, dict[str, Any], BaseModel, Exception]) -> None:
        """Queue a response or Exception for the next generate call."""
        self._response_queue.append(response)

    def set_default_response(self, response: Optional[Union[str, dict[str, Any], BaseModel, Exception]]) -> None:
        """Set a persistent default response when queue is empty."""
        self._default_response = response

    def clear(self) -> None:
        """Clear call history and queued responses."""
        self._response_queue.clear()
        self.call_history.clear()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> tuple[Union[str, BaseModel], LLMCallMetadata]:
        """Produce a deterministic mock response without making external network calls."""
        call_record = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_model": response_model.__name__,
            "timestamp": time.time(),
        }
        self.call_history.append(call_record)

        # 1. Fetch next response from queue or default
        if self._response_queue:
            raw_response = self._response_queue.popleft()
        elif self._default_response is not None:
            raw_response = self._default_response
        else:
            raw_response = self._build_deterministic_default(response_model)

        # 2. Check if an exception was injected
        if isinstance(raw_response, Exception):
            raise raw_response

        metadata = LLMCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=self._latency_ms,
            input_tokens=max(1, len(user_prompt.split())),
            output_tokens=25,
            total_tokens=max(1, len(user_prompt.split())) + 25,
        )

        # 3. Format response
        if isinstance(raw_response, BaseModel):
            return raw_response, metadata
        elif isinstance(raw_response, dict):
            return json.dumps(raw_response), metadata
        elif isinstance(raw_response, str):
            return raw_response, metadata
        else:
            raise LLMProviderError(f"Unsupported mock response type: {type(raw_response)}")

    def _build_deterministic_default(self, response_model: Type[BaseModel]) -> Union[str, BaseModel]:
        """Construct sensible default responses for standard project schemas."""
        if response_model is ConnectivityResponse:
            return ConnectivityResponse(status="ok")

        if response_model is ClaimExtractionResult:
            return ClaimExtractionResult(
                claims=[
                    Claim(
                        id=1,
                        claim_index=1,
                        text="Deterministic mock claim extracted from source text.",
                        checkable=True,
                        source_span="Deterministic mock claim extracted from source text.",
                    )
                ],
                source_text_length=50,
                truncated=False,
            )

        if response_model is LLMAssessment:
            return LLMAssessment(
                label=VerdictLabel.likely_reliable,
                source_attribution=SourceAttribution.strong,
                evidence_support=EvidenceSupport.supported,
                internal_consistency=InternalConsistency.consistent,
                rhetorical_cues=[],
                unsupported_claims=[],
                short_reason="Deterministic mock verification completed successfully.",
            )

        raise LLMProviderError(
            f"MockLLMProvider has no predefined response for schema '{response_model.__name__}'. "
            "Please queue a response using mock_provider.queue_response()."
        )
