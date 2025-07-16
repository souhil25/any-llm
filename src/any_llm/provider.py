# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
import importlib
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Type, Union

from openai.types.chat.chat_completion import ChatCompletion

from pydantic import BaseModel

from any_llm.exceptions import UnsupportedProviderError


class ProviderName(str, Enum):
    """String enum for supported providers."""

    ANTHROPIC = "anthropic"
    AWS = "aws"
    AZURE = "azure"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GOOGLE = "google"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    INCEPTION = "inception"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    OLLAMA = "ollama"
    OPENAI = "openai"
    SAMBANOVA = "sambanova"
    TOGETHER = "together"
    WATSONX = "watsonx"
    XAI = "xai"


class ApiConfig(BaseModel):
    """Configuration for the provider."""

    api_key: str | None = None
    api_base: str | None = None


class Provider(ABC):
    """Provider for the LLM."""

    def __init__(self, config: ApiConfig) -> None:
        self.config = config

    @abstractmethod
    def completion(self, model: str, messages: list[dict[str, Any]], **kwargs: dict[str, Any]) -> ChatCompletion:
        """Must be implemented by each provider."""
        raise NotImplementedError


class ProviderFactory:
    """Factory to dynamically load provider instances based on the naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: Union[str, ProviderName], config: ApiConfig) -> Provider:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert to string if it's a ProviderName enum
        if isinstance(provider_key, ProviderName):
            provider_key = provider_key.value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            raise ImportError(msg) from e

        # Instantiate the provider class
        provider_class: Type[Provider] = getattr(module, provider_class_name)
        return provider_class(config=config)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider keys."""
        # Return the enum values as strings
        return [provider.value for provider in ProviderName]

    @classmethod
    def get_provider_enum(cls, provider_key: str) -> ProviderName:
        """Convert a string provider key to a ProviderName enum."""
        try:
            return ProviderName(provider_key)
        except ValueError:
            supported = [provider.value for provider in ProviderName]
            raise UnsupportedProviderError(provider_key, supported)
