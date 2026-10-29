"""Provider factory for instantiating LLM provider adapters.

Supports provider swapping based on environment configuration or injected parameters.
"""

from typing import Optional

from backend.app.ai.exceptions import LLMConfigurationError
from backend.app.ai.provider_base import BaseLLMProvider
from backend.app.ai.providers.groq_provider import GroqProvider
from backend.app.ai.providers.mock_provider import MockLLMProvider
from backend.app.config import LLMConfig

SUPPORTED_PROVIDERS: tuple[str, ...] = ("mock", "groq")
POTENTIAL_LIVE_PROVIDERS: tuple[str, ...] = ("openai", "anthropic")


def get_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    """Instantiate and return the configured LLM provider adapter.

    Args:
        config: Optional LLMConfig instance. If None, loaded from environment.

    Returns:
        Configured BaseLLMProvider implementation.

    Raises:
        LLMConfigurationError: If provider is unspecified, unknown, or missing required credentials.
    """
    if config is None:
        config = LLMConfig.from_env()

    if not config.provider or not config.provider.strip():
        raise LLMConfigurationError(
            "LLM provider is not configured. Set LLM_PROVIDER in your environment or .env file."
        )

    provider_name = config.provider.strip().lower()

    if provider_name == "mock":
        model_name = config.model.strip() if config.model and config.model.strip() else "mock-model-v1"
        return MockLLMProvider(model_name=model_name)

    if provider_name == "groq":
        if not config.api_key or not config.api_key.strip():
            raise LLMConfigurationError(
                "Missing API key for provider 'groq'. Set GROQ_API_KEY in your environment."
            )
        if not config.model or not config.model.strip():
            raise LLMConfigurationError(
                "Missing model for provider 'groq'. Set GROQ_MODEL in your environment."
            )
        return GroqProvider(
            api_key=config.api_key,
            model_name=config.model,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
        )

    if provider_name == "gemini":
        raise LLMConfigurationError("V2 runtime is configured for Groq-only operation.")

    if provider_name in POTENTIAL_LIVE_PROVIDERS:
        if not config.api_key or not config.api_key.strip():
            raise LLMConfigurationError(
                f"Missing API key for provider '{provider_name}'. Set LLM_API_KEY in your environment."
            )
        raise LLMConfigurationError(
            f"Provider '{provider_name}' adapter is not implemented or its SDK is not installed in the environment."
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider '{provider_name}'. Supported providers: {list(SUPPORTED_PROVIDERS)}."
    )
