# Inspired by https://github.com/andrewyng/aisuite/tree/main/aisuite
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion


class ApiConfig(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None


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

    PROVIDERS_DIR = Path(__file__).parent.parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: str, config: ApiConfig) -> Provider:
        """Dynamically load and create an instance of a provider based on the naming convention."""
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
        # list all subdirectories in the providers directory
        return [d.name for d in cls.PROVIDERS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
