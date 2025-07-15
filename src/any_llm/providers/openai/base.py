import os
from typing import Any
from abc import ABC

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from any_llm.provider import Provider, ApiConfig


class BaseOpenAIProvider(Provider, ABC):
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

    def __init__(self, config: ApiConfig) -> None:
        """Initialize OpenAI-compatible provider."""
        super().__init__(config)

        client_kwargs: dict[str, Any] = {}

        if not config.api_base:
            client_kwargs["base_url"] = self.DEFAULT_API_BASE
        else:
            client_kwargs["base_url"] = config.api_base

        if not config.api_key and not os.getenv(self.ENV_API_KEY_NAME):
            msg = f"No {self.PROVIDER_NAME} API key provided. Please provide it in the config or set the {self.ENV_API_KEY_NAME} environment variable."
            raise ValueError(msg)

        # Get API key from environment if not provided in config
        api_key = config.api_key or os.getenv(self.ENV_API_KEY_NAME)
        if api_key is None:
            msg = f"No {self.PROVIDER_NAME} API key provided. Please provide it in the config or set the {self.ENV_API_KEY_NAME} environment variable."
            raise ValueError(msg)

        client_kwargs["api_key"] = api_key

        # Create the OpenAI client
        self.client = OpenAI(**client_kwargs)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using OpenAI-compatible API."""
        if "response_format" in kwargs:
            response: ChatCompletion = self.client.chat.completions.parse(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

        return response
