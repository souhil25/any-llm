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
from any_llm.utils import convert_response_to_openai
from any_llm.provider import Provider, ApiConfig


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Mistral."""
    if "response_format" in kwargs and issubclass(
        kwargs["response_format"],
        BaseModel,
    ):
        kwargs["response_format"] = response_format_from_pydantic_model(
            kwargs["response_format"],
        )
    return kwargs


class MistralProvider(Provider):
    def __init__(self, config: ApiConfig) -> None:
        """Initialize Mistral provider."""
        if not config.api_key:
            config.api_key = os.getenv("MISTRAL_API_KEY")
        if not config.api_key:
            msg = "No Mistral API key provided. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            raise ValueError(msg)
        self.client = Mistral(api_key=config.api_key, server_url=config.api_base)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Mistral."""
        kwargs = _convert_kwargs(kwargs)

        # Make the request to Mistral
        response = self.client.chat.complete(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        return convert_response_to_openai(response.model_dump())
