"""Official Groq Provider Adapter (V2.1 Single-Provider Runtime).

Communicates with Groq models using the official groq Python SDK.
Translates structured Pydantic models into strict JSON Schema contracts and maps
all downstream SDK errors into the project's provider-neutral exception hierarchy.
Security guarantee: API keys are masked from exception messages and never logged.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Optional, Type, Union

import groq
from pydantic import BaseModel

from backend.app.ai.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
    sanitize_sensitive_text,
)
from backend.app.ai.provider_base import BaseLLMProvider
from backend.app.schemas import LLMCallMetadata

DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-20b"


def _normalize_schema_for_strict(schema_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Pydantic v2 JSON Schema to satisfy Groq strict json_schema constraints.

    Groq's strict structured-output mode requires:
    1. Every object schema must have additionalProperties set to false.
    2. All properties listed in 'properties' must be included in 'required'.
    3. Rules apply recursively to root, nested objects, array items, and $defs.
    4. anyOf branches must be disambiguated: inlines $ref inside anyOf to expose enum/type discriminators.
    """
    normalized = copy.deepcopy(schema_dict)
    defs = normalized.get("$defs") or normalized.get("definitions") or {}

    def _traverse(node: Any) -> None:
        if not isinstance(node, dict):
            return

        # Inline $ref inside anyOf branches so Groq sees discriminators/enums directly
        if "anyOf" in node and isinstance(node["anyOf"], list):
            new_any_of = []
            for sub in node["anyOf"]:
                if isinstance(sub, dict) and "$ref" in sub and not any(k in sub for k in ("enum", "type", "const")):
                    ref_key = sub["$ref"].split("/")[-1]
                    if ref_key in defs:
                        resolved = copy.deepcopy(defs[ref_key])
                        new_any_of.append(resolved)
                        continue
                new_any_of.append(sub)
            node["anyOf"] = new_any_of

        # If this node represents an object type or contains properties
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            if "properties" in node and isinstance(node["properties"], dict):
                # Under strict: true, all property names must appear in required
                node["required"] = list(node["properties"].keys())
                for prop_val in node["properties"].values():
                    _traverse(prop_val)

        if "items" in node:
            _traverse(node["items"])

        # Traverse definition dictionaries ($defs or definitions)
        for def_key in ("$defs", "definitions"):
            if def_key in node and isinstance(node[def_key], dict):
                for def_val in node[def_key].values():
                    _traverse(def_val)

        # Traverse composite schemas
        for comp_key in ("anyOf", "allOf", "oneOf"):
            if comp_key in node and isinstance(node[comp_key], list):
                for sub_schema in node[comp_key]:
                    _traverse(sub_schema)

    _traverse(normalized)
    return normalized


class GroqProvider(BaseLLMProvider):
    """Provider adapter for Groq via the official groq Python SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        client: Optional[Any] = None,
        max_completion_tokens: int = 2048,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise LLMConfigurationError(
                "Groq API key is missing or empty. Set GROQ_API_KEY in your environment or .env file."
            )

        resolved_model = model_name.strip() if model_name and str(model_name).strip() else DEFAULT_GROQ_MODEL
        if not resolved_model:
            raise LLMConfigurationError(
                "Groq model name is missing or empty. Set GROQ_MODEL in your environment or .env file."
            )

        self._api_key = str(api_key).strip()
        self._model_name = resolved_model
        self._timeout_seconds = float(timeout_seconds)
        self._temperature = float(temperature)
        self._max_completion_tokens = int(max_completion_tokens) if max_completion_tokens is not None else 2048

        # Allow client injection for testing with zero network calls
        if client is not None:
            self._client = client
        else:
            self._client = groq.Groq(
                api_key=self._api_key,
                timeout=self._timeout_seconds,
                max_retries=0,  # Retries are managed by LLMClient to avoid nested loops
            )

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> tuple[Union[str, BaseModel], LLMCallMetadata]:
        """Invoke Groq API requesting structured JSON conforming to response_model."""
        start_time = time.perf_counter()
        raw_text: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        total_tokens: Optional[int] = None

        raw_schema = response_model.model_json_schema()
        normalized_schema = _normalize_schema_for_strict(raw_schema)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": normalized_schema,
            },
        }

        call_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "response_format": response_format,
            "temperature": self._temperature,
        }
        if self._max_completion_tokens is not None:
            call_kwargs["max_completion_tokens"] = self._max_completion_tokens

        try:
            completion = self._client.chat.completions.create(**call_kwargs)

            # Extract raw response text
            if not getattr(completion, "choices", None) or len(completion.choices) == 0:
                raise LLMResponseValidationError("Groq returned empty response choices.")

            choice = completion.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                raise LLMResponseValidationError("Groq response choice missing message object.")

            raw_text = getattr(message, "content", None)
            if raw_text is None or not str(raw_text).strip():
                raise LLMResponseValidationError("Groq returned an empty response.")

            # Capture usage telemetry if provided by the SDK/API
            usage = getattr(completion, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

        except groq.AuthenticationError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMConfigurationError(
                f"Groq authentication failed (HTTP 401): {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.RateLimitError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMProviderError(
                f"Groq rate limit or quota exhausted (HTTP 429): {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.APITimeoutError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMTimeoutError(
                f"Groq request timed out after {self._timeout_seconds}s: {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.APIConnectionError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMProviderError(
                f"Groq network connection error: {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.NotFoundError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMConfigurationError(
                f"Groq model or endpoint not found (HTTP 404): {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.BadRequestError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            msg_lower = str(exc).lower()
            if "model" in msg_lower:
                raise LLMConfigurationError(
                    f"Groq model configuration error: {sanitized}",
                    secret_to_mask=self._api_key,
                ) from exc
            elif "schema" in msg_lower or "json" in msg_lower:
                raise LLMResponseValidationError(
                    f"Groq structured schema rejected by API: {sanitized}",
                    secret_to_mask=self._api_key,
                ) from exc
            else:
                raise LLMProviderError(
                    f"Groq bad request (HTTP 400): {sanitized}",
                    secret_to_mask=self._api_key,
                ) from exc

        except groq.InternalServerError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMProviderError(
                f"Groq internal server error (HTTP 5xx): {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.APIStatusError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            status_code = getattr(exc, "status_code", "unknown")
            raise LLMProviderError(
                f"Groq API status error (HTTP {status_code}): {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except groq.APIError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMProviderError(
                f"Groq API error: {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except (TimeoutError, TimeoutException) if "TimeoutException" in globals() else TimeoutError as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMTimeoutError(
                f"Groq invocation timed out: {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        except Exception as exc:
            sanitized = sanitize_sensitive_text(str(exc), self._api_key)
            raise LLMProviderError(
                f"Groq unexpected error: {sanitized}",
                secret_to_mask=self._api_key,
            ) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = LLMCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=round(elapsed_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return raw_text, metadata
