import os
from typing import Any

try:
    import together
    from together.types import (
        ChatCompletionResponse,
    )
except ImportError:
    msg = "together is not installed. Please install it with `pip install any-llm-sdk[together]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.base_framework import create_completion_from_response


class TogetherProvider(Provider):
    """
    Together AI Provider implementation using the official Together AI SDK.

    This provider connects to Together AI's API using the Together SDK.
    It supports structured outputs (JSON schema, regex) as documented in Together AI's API.

    Configuration:
    - api_key: Together AI API key (can be set via TOGETHER_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to Together AI's API)

    Example usage:
        config = ApiConfig(api_key="your-together-api-key")
        provider = TogetherProvider(config)
        response = provider.completion("meta-llama/Llama-3.1-8B-Instruct-Turbo", messages=[...])
    """

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Together AI provider."""
        if not config.api_key:
            config.api_key = os.getenv("TOGETHER_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Together AI", "TOGETHER_API_KEY")

        # Initialize Together client
        if config.api_base:
            self.client = together.Together(api_key=config.api_key, base_url=config.api_base)
        else:
            self.client = together.Together(api_key=config.api_key)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Together AI."""

        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to Fireworks JSON schema format
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
                }
            else:
                # response_format is already a dict, pass it through
                kwargs["response_format"] = response_format

        # Make the API call. Since streaming is not supported, this won't be an iter
        response: ChatCompletionResponse = self.client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,
            **kwargs,
        )

        # Convert to OpenAI format using the utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            model=model,
            provider_name="together",
            finish_reason_mapping={
                "stop": "stop",
                "length": "length",
                "tool_calls": "tool_calls",
                "content_filter": "content_filter",
            },
            token_field_mapping={
                "prompt_tokens": "prompt_tokens",
                "completion_tokens": "completion_tokens",
                "total_tokens": "total_tokens",
            },
        )
