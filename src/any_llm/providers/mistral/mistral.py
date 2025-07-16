import os
from typing import Any

try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.base_framework import create_completion_from_response
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class MistralProvider(Provider):
    """Mistral Provider using the new response conversion utilities."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Mistral provider."""
        if not config.api_key:
            config.api_key = os.getenv("MISTRAL_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Mistral", "MISTRAL_API_KEY")

        self.client = Mistral(api_key=config.api_key, server_url=config.api_base)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Mistral."""

        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", "Mistral")

        # Handle response_format for Pydantic models
        if "response_format" in kwargs and issubclass(kwargs["response_format"], BaseModel):
            kwargs["response_format"] = response_format_from_pydantic_model(kwargs["response_format"])

        # Make the API call
        response = self.client.chat.complete(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            model=model,
            provider_name="mistral",
        )
