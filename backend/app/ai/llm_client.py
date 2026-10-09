"""High-level LLM Client and structured-output orchestration adapter.

All AI services in the project communicate with LLMs solely through this interface.
No business logic module directly interacts with external LLM provider SDKs.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import time
from typing import Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from backend.app.ai.exceptions import (
    LLMBaseError,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
    sanitize_sensitive_text,
)
from backend.app.ai.provider_base import BaseLLMProvider, LLMProvider
from backend.app.ai.provider_factory import get_provider
from backend.app.config import LLMConfig
from backend.app.schemas import LLMCallMetadata

T = TypeVar("T", bound=BaseModel)


class LLMClient(LLMProvider):
    """Orchestrates structured LLM interaction with validation, timeouts, and safe retries."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        self.config: LLMConfig = config or LLMConfig.from_env()
        self.provider: BaseLLMProvider = provider if provider is not None else get_provider(self.config)
        self.last_metadata: Optional[LLMCallMetadata] = None

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider='{self.provider.provider_name}', "
            f"model='{self.provider.model_name}', "
            f"timeout={self.config.timeout_seconds}s, "
            f"max_retries={self.config.max_retries})"
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:
        """Execute structured LLM invocation and return validated Pydantic model.

        Args:
            system_prompt: Guiding instructions for the model.
            user_prompt: Specific task or query content.
            response_model: Expected Pydantic model class for response validation.

        Returns:
            Validated Pydantic model instance.

        Raises:
            LLMConfigurationError: For configuration or authentication failures.
            LLMTimeoutError: If the call exceeds configured timeout.
            LLMResponseValidationError: If output is invalid JSON or violates schema.
            LLMProviderError: For downstream provider communication failures.
        """
        model_instance, metadata = self.generate_structured_with_metadata(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )
        return model_instance

    def generate_structured_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> tuple[T, LLMCallMetadata]:
        """Execute structured invocation and return both validated model and telemetry metadata."""
        max_attempts = max(1, 1 + self.config.max_retries)
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                raw_response, metadata = self._invoke_with_timeout(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                )
                validated_model = self._validate_response(raw_response, response_model)
                self.last_metadata = metadata
                return validated_model, metadata

            except LLMConfigurationError:
                # Configuration / auth errors must NOT be retried.
                raise

            except LLMResponseValidationError:
                # Schema validation errors are deterministic given the same response; do not retry.
                raise

            except LLMTimeoutError as err:
                last_error = err
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))

            except LLMProviderError as err:
                last_error = err
                if attempt == max_attempts - 1 or "retry in" in str(err).lower() or "resource_exhausted" in str(err).lower():
                    raise
                time.sleep(2.0 * (attempt + 1))

            except Exception as err:
                sanitized_msg = sanitize_sensitive_text(str(err), self.config.api_key)
                wrapped_err = LLMProviderError(f"Provider invocation failed: {sanitized_msg}")
                last_error = wrapped_err
                if attempt == max_attempts - 1:
                    raise wrapped_err from err
                time.sleep(2.0 * (attempt + 1))

        if last_error:
            raise last_error
        raise LLMProviderError("Execution failed without producing a response or error.")

    def _invoke_with_timeout(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> tuple[Union[str, BaseModel], LLMCallMetadata]:
        """Wrap provider generation in a strict timeout."""
        timeout_sec = self.config.timeout_seconds

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                self.provider.generate,
                system_prompt,
                user_prompt,
                response_model,
            )
            return future.result(timeout=timeout_sec)
        except (FutureTimeoutError, TimeoutError) as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise LLMTimeoutError(
                f"LLM request timed out after {timeout_sec} seconds.",
                secret_to_mask=self.config.api_key,
            ) from exc
        except LLMBaseError:
            executor.shutdown(wait=False)
            raise
        except Exception as exc:
            executor.shutdown(wait=False)
            sanitized = sanitize_sensitive_text(str(exc), self.config.api_key)
            raise LLMProviderError(
                f"Provider raised an unhandled exception: {sanitized}",
                secret_to_mask=self.config.api_key,
            ) from exc
        finally:
            executor.shutdown(wait=False)

    def _validate_response(
        self,
        raw_response: Union[str, BaseModel, dict[str, Any]],
        response_model: Type[T],
    ) -> T:
        """Parse JSON and validate strictly against the expected Pydantic model."""
        if isinstance(raw_response, response_model):
            return raw_response

        if isinstance(raw_response, BaseModel):
            raise LLMResponseValidationError(
                f"Expected response model '{response_model.__name__}', "
                f"got '{type(raw_response).__name__}'."
            )

        if isinstance(raw_response, dict):
            try:
                return response_model.model_validate(raw_response)
            except Exception as exc:
                sanitized = sanitize_sensitive_text(str(exc), self.config.api_key)
                raise LLMResponseValidationError(
                    f"Response failed schema validation for '{response_model.__name__}': {sanitized}"
                ) from exc

        if isinstance(raw_response, str):
            stripped = raw_response.strip()
            if not stripped:
                raise LLMResponseValidationError(
                    f"Received empty response from LLM; expected '{response_model.__name__}' JSON."
                )

            try:
                parsed_json = json.loads(stripped)
            except json.JSONDecodeError as exc:
                sanitized = sanitize_sensitive_text(str(exc), self.config.api_key)
                raise LLMResponseValidationError(
                    f"Response is not valid JSON: {sanitized}"
                ) from exc

            if not isinstance(parsed_json, dict):
                raise LLMResponseValidationError(
                    f"Expected JSON object for '{response_model.__name__}', got {type(parsed_json).__name__}."
                )

            try:
                return response_model.model_validate(parsed_json)
            except Exception as exc:
                sanitized = sanitize_sensitive_text(str(exc), self.config.api_key)
                raise LLMResponseValidationError(
                    f"Response failed schema validation for '{response_model.__name__}': {sanitized}"
                ) from exc

        raise LLMResponseValidationError(
            f"Unsupported raw response type '{type(raw_response).__name__}' for '{response_model.__name__}'."
        )
