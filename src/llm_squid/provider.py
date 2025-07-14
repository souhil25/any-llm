import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message: str) -> None:
        """Initialize the LLMError."""
        super().__init__(message)


class Provider(ABC):
    """Provider for the LLM."""

    @abstractmethod
    def chat_completions_create(
        self, model: str, messages: list[Any], **kwargs: Any
    ) -> Any:
        """Abstract method for chat completion calls, to be implemented by each provider."""


class ProviderFactory:
    """Factory to dynamically load provider instances based on the naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: str, config: dict[str, Any]) -> Any:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"llm_squid.{provider_module_name}"

        # Lazily load the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            raise ImportError(msg) from e

        # Instantiate the provider class
        provider_class = getattr(module, provider_class_name)
        return provider_class(**config)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider keys."""
        # This is a placeholder - implement based on your actual provider discovery logic
        return ["mistral"]  # Add other providers as needed
