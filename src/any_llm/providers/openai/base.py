from typing import Any
from abc import ABC

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from any_llm.provider import ApiConfig
from any_llm.providers.base_framework import BaseProviderFramework


class BaseOpenAIProvider(BaseProviderFramework, ABC):
    """
    Base provider for OpenAI-compatible services.

    This class provides a common foundation for providers that use OpenAI-compatible APIs.
    Subclasses only need to override configuration defaults and client initialization
    if needed.
    """

    # Default configuration values that can be overridden by subclasses
    DEFAULT_API_BASE: str
    ENV_API_KEY_NAME: str
    PROVIDER_NAME: str

    def _initialize_client(self, config: ApiConfig) -> None:
        """Initialize OpenAI-compatible client."""
        client_kwargs: dict[str, Any] = {}

        if not config.api_base:
            client_kwargs["base_url"] = self.DEFAULT_API_BASE
        else:
            client_kwargs["base_url"] = config.api_base

        # API key is already validated in BaseProviderFramework
        client_kwargs["api_key"] = config.api_key

        # Create the OpenAI client
        self.client = OpenAI(**client_kwargs)

    def _convert_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert kwargs for OpenAI-compatible providers (minimal conversion needed)."""
        return kwargs

    def _make_api_call(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
        """Make the API call to OpenAI-compatible service."""
        if "response_format" in kwargs:
            response: ChatCompletion = self.client.chat.completions.parse(  # type: ignore[attr-defined]
                model=model,
                messages=messages,
                **kwargs,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
        return response

    def _convert_response(self, raw_response: ChatCompletion) -> ChatCompletion:
        """Convert response for OpenAI-compatible providers (no conversion needed)."""
        return raw_response
