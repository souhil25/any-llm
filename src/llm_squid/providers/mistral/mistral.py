import os
from typing import Any, cast

from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from llm_squid.utils import convert_request_to_openai, convert_response_to_openai
from llm_squid.utils.provider import Provider


class MistralProvider(Provider):
    def __init__(self, **config: Any) -> None:
        """Initialize Mistral provider."""
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            msg = "Mistral API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            raise ValueError(msg)
        self.client = Mistral(**config)

    def convert_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Format the kwargs for Mistral."""
        if "response_format" in kwargs and issubclass(
            kwargs["response_format"],
            BaseModel,
        ):
            kwargs["response_format"] = response_format_from_pydantic_model(
                kwargs["response_format"],
            )
        return kwargs

    def completion(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Mistral."""
        kwargs = self.convert_kwargs(kwargs)
        # Transform messages using converter
        transformed_messages = convert_request_to_openai(messages)

        # Make the request to Mistral
        # Cast to Any to avoid type issues since Mistral accepts dict format
        response = self.client.chat.complete(
            model=model,
            messages=transformed_messages,
            **kwargs,
        )

        return convert_response_to_openai(response.model_dump())
