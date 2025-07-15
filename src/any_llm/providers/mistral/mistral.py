import os
from typing import Any

from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.utils import convert_response_to_openai
from any_llm.utils.provider import Provider


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
    def __init__(self, **config: Any) -> None:
        """Initialize Mistral provider."""
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            msg = "Mistral API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            raise ValueError(msg)
        self.client = Mistral(**config)

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
