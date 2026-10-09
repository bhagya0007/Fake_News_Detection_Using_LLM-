"""Official Google Gemini Provider Adapter (Step 10.5).

Communicates with Gemini models using the official google-genai SDK.
Translates structured Pydantic models into JSON Schema contracts and maps
all downstream SDK errors into the project's exception hierarchy.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Type, Union

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from backend.app.ai.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from backend.app.ai.provider_base import BaseLLMProvider
from backend.app.schemas import LLMCallMetadata


def _sanitize_schema(schema_obj: Any) -> Any:
    """Recursively strip unsupported attributes from google.genai Schema for API compatibility."""
    if hasattr(schema_obj, "additional_properties"):
        schema_obj.additional_properties = None
    if hasattr(schema_obj, "properties") and schema_obj.properties:
        for prop in schema_obj.properties.values():
            _sanitize_schema(prop)
    if hasattr(schema_obj, "items") and schema_obj.items:
        _sanitize_schema(schema_obj.items)
    return schema_obj


def _prepare_response_schema(response_model: Type[BaseModel]) -> Any:
    """Transform Pydantic model into Gemini-compatible OpenAPI schema object."""
    try:
        from google.genai._transformers import t_schema
        schema = t_schema(None, response_model)
        return _sanitize_schema(schema)
    except Exception:
        return response_model


class GeminiProvider(BaseLLMProvider):
    """Provider adapter for Google Gemini via the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        client: Optional[Any] = None,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise LLMConfigurationError(
                "Gemini API key is missing or empty. Set LLM_API_KEY in your environment or .env file."
            )
        if not model_name or not str(model_name).strip():
            raise LLMConfigurationError(
                "Gemini model name is missing or empty. Set LLM_MODEL in your environment or .env file."
            )

        self._api_key = str(api_key).strip()
        self._model_name = str(model_name).strip()
        self._timeout_seconds = float(timeout_seconds)
        self._temperature = float(temperature)

        # Allow client injection for unit tests without network calls
        if client is not None:
            self._client = client
        else:
            timeout_ms = max(1000, int(self._timeout_seconds * 1000))
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> tuple[Union[str, BaseModel], LLMCallMetadata]:
        """Invoke Gemini API requesting structured JSON conforming to response_model."""
        start_time = time.perf_counter()
        raw_text: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        total_tokens: Optional[int] = None

        try:
            # 1. Official Gemini Generative Models API with native structured output
            if hasattr(self._client, "models") and hasattr(self._client.models, "generate_content"):
                prepared_schema = _prepare_response_schema(response_model)
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=prepared_schema,
                    system_instruction=system_prompt,
                    temperature=self._temperature,
                )
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_prompt,
                    config=config,
                )
                raw_text = getattr(response, "text", None)
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta is not None:
                    input_tokens = getattr(usage_meta, "prompt_token_count", None)
                    output_tokens = getattr(usage_meta, "candidates_token_count", None)
                    total_tokens = getattr(usage_meta, "total_token_count", None)

            # 2. Alternative Interactions API support
            elif hasattr(self._client, "interactions") and hasattr(self._client.interactions, "create"):
                create_kwargs = {
                    "model": self._model_name,
                    "system_instruction": system_prompt,
                    "input": user_prompt,
                    "response_format": {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": response_model.model_json_schema(),
                    },
                }
                if self._temperature is not None:
                    create_kwargs["generation_config"] = {"temperature": self._temperature}

                interaction = self._client.interactions.create(**create_kwargs)

                if hasattr(interaction, "output_text") and interaction.output_text:
                    raw_text = interaction.output_text

                usage = getattr(interaction, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "total_input_tokens", None)
                    output_tokens = getattr(usage, "total_output_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)

        except errors.APIError as exc:
            err_code = getattr(exc, "code", 500)
            err_status = getattr(exc, "status", "") or ""
            err_msg = getattr(exc, "message", str(exc)) or str(exc)

            if err_code in (401, 403) or "unauthenticated" in err_msg.lower() or "permission_denied" in err_msg.lower():
                raise LLMConfigurationError(
                    f"Gemini authentication failed ({err_code} {err_status}): {err_msg}",
                    secret_to_mask=self._api_key,
                ) from exc
            elif err_code == 404 or "not_found" in err_msg.lower():
                raise LLMConfigurationError(
                    f"Gemini model '{self._model_name}' not found or unavailable ({err_code} {err_status}): {err_msg}",
                    secret_to_mask=self._api_key,
                ) from exc
            elif err_code == 408 or "deadline" in err_msg.lower() or "timed out" in err_msg.lower():
                raise LLMTimeoutError(
                    f"Gemini request timed out ({err_code}): {err_msg}",
                    secret_to_mask=self._api_key,
                ) from exc
            else:
                raise LLMProviderError(
                    f"Gemini provider error ({err_code} {err_status}): {err_msg}",
                    secret_to_mask=self._api_key,
                ) from exc

        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out: {exc}",
                secret_to_mask=self._api_key,
            ) from exc

        except Exception as exc:
            raise LLMProviderError(
                f"Gemini execution error: {exc}",
                secret_to_mask=self._api_key,
            ) from exc

        if not raw_text or not raw_text.strip():
            raise LLMResponseValidationError(
                f"Gemini returned an empty response for model '{self._model_name}'.",
                secret_to_mask=self._api_key,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = LLMCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return raw_text, metadata
