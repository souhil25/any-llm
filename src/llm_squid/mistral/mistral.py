import os
from typing import Any, cast

from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from llm_squid.message import ChatCompletionResponse
from llm_squid.message_converter import OpenAICompliantMessageConverter
from llm_squid.provider import LLMError, Provider


class MistralMessageConverter(OpenAICompliantMessageConverter):
    def convert_response(self, response_data: Any) -> ChatCompletionResponse:
        """Convert Mistral's response to our standard format."""
        # Convert Mistral's response object to dict format
        response_dict = response_data.model_dump()
        return super().convert_response(response_dict)


class MistralProvider(Provider):
    def __init__(self, **config: Any) -> None:
        """Initialize Mistral provider."""
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            msg = "Mistral API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            raise ValueError(msg)
        self.client = Mistral(**config)
        self.transformer = MistralMessageConverter()

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

    def chat_completions_create(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion using Mistral."""
        kwargs = self.convert_kwargs(kwargs)
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            # Make the request to Mistral
            # Cast to Any to avoid type issues since Mistral accepts dict format
            response = self.client.chat.complete(
                model=model,
                messages=cast("Any", transformed_messages),
                **kwargs,
            )

            return self.transformer.convert_response(response)
        except Exception as e:
            msg = f"An error occurred: {e}"
            raise LLMError(msg) from e
