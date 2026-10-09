"""AI / LLM Adapter Package.

Provides a unified, secure, structured interface for all LLM interactions in the project.
No downstream service should import third-party LLM SDKs directly.
"""

from backend.app.ai.exceptions import (
    LLMBaseError,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from backend.app.ai.llm_client import LLMClient
from backend.app.ai.provider_base import BaseLLMProvider, LLMProvider
from backend.app.ai.provider_factory import get_provider
from backend.app.ai.providers.gemini_provider import GeminiProvider
from backend.app.ai.providers.mock_provider import MockLLMProvider

__all__ = [
    "LLMClient",
    "LLMProvider",
    "BaseLLMProvider",
    "GeminiProvider",
    "MockLLMProvider",
    "get_provider",
    "LLMBaseError",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMResponseValidationError",
]
