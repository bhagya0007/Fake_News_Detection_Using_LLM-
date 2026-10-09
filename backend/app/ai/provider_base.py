"""Base contracts for LLM providers and adapters.

Defines both the abstract base class for provider implementations and
the runtime protocol for structured LLM interactions.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Type, TypeVar, Union, runtime_checkable

from pydantic import BaseModel

from backend.app.schemas import LLMCallMetadata

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the public interface for structured LLM execution."""

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:
        """Generate a validated structured Pydantic object based on prompts and schema."""
        ...


class BaseLLMProvider(ABC):
    """Abstract base class that all concrete LLM provider adapters must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g. 'mock', 'gemini', 'openai')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier (e.g. 'mock-model-v1')."""
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
    ) -> tuple[Union[str, BaseModel], LLMCallMetadata]:
        """Invoke underlying provider and return (raw JSON or model instance, metadata).

        Concrete implementations must not swallow authentication or timeout exceptions;
        they should allow LLMClient or the provider implementation to raise
        appropriate LLMException types.
        """
        pass
