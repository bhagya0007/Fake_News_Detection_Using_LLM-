"""Exception hierarchy for the LLM client and provider adapter layer.

Security guarantee:
Exception messages and representations must NEVER expose secret credentials,
API keys, or authorization tokens.
"""

from typing import Optional


def sanitize_sensitive_text(text: str, secret: Optional[str] = None) -> str:
    """Sanitize potential secret leaks from exception messages."""
    if not text:
        return text
    if secret and secret in text:
        text = text.replace(secret, "[REDACTED_API_KEY]")
    return text


class LLMBaseError(Exception):
    """Base exception for all LLM client and provider operations."""

    def __init__(self, message: str = "", secret_to_mask: Optional[str] = None) -> None:
        self.raw_message = message
        sanitized = sanitize_sensitive_text(message, secret_to_mask)
        super().__init__(sanitized)


class LLMConfigurationError(LLMBaseError):
    """Raised when provider configuration or credentials are missing or invalid."""

    pass


class LLMProviderError(LLMBaseError):
    """Raised when an LLM provider fails during invocation (API error, rate limit, server failure)."""

    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM call exceeds the configured timeout duration."""

    pass


class LLMResponseValidationError(LLMBaseError):
    """Raised when an LLM response cannot be parsed as valid JSON or fails Pydantic schema validation."""

    pass
