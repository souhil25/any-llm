import os
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    msg = "huggingface-hub is not installed. Please install it with `pip install any-llm-sdk[huggingface]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.base_framework import (
    create_completion_from_response,
    remove_unsupported_params,
)


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the new response conversion utilities."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize HuggingFace provider."""
        if not config.api_key:
            config.api_key = os.getenv("HF_TOKEN")
        if not config.api_key:
            raise MissingApiKeyError("HuggingFace", "HF_TOKEN")

        self.client = InferenceClient(token=config.api_key, timeout=30)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using HuggingFace."""
        # Convert max_tokens to max_new_tokens (HuggingFace specific)
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        # Remove unsupported parameters
        kwargs = remove_unsupported_params(kwargs, ["response_format", "parallel_tool_calls"])

        # Ensure message content is always a string and handle tool calls
        cleaned_messages = []
        for message in messages:
            cleaned_message = {
                "role": message["role"],
                "content": message.get("content") or "",
            }

            # Handle tool calls if present
            if "tool_calls" in message and message["tool_calls"]:
                cleaned_message["tool_calls"] = message["tool_calls"]

            # Handle tool call ID for tool messages
            if "tool_call_id" in message:
                cleaned_message["tool_call_id"] = message["tool_call_id"]

            cleaned_messages.append(cleaned_message)

        try:
            # Make the API call
            response = self.client.chat_completion(
                model=model,
                messages=cleaned_messages,
                **kwargs,
            )

            # Convert to OpenAI format using the new utility
            return create_completion_from_response(
                response_data=response,
                model=model,
                provider_name="huggingface",
            )

        except Exception as e:
            raise RuntimeError(f"HuggingFace API error: {e}") from e
