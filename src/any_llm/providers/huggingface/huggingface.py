import os
import json
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    msg = "huggingface-hub is not installed. Please install it with `pip install any-llm-sdk[huggingface]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.base_framework import (
    create_completion_from_response,
    remove_unsupported_params,
)


def _convert_pydantic_to_huggingface_json(
    pydantic_model: type[BaseModel], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert Pydantic model to HuggingFace-compatible JSON instructions.

    Following a similar pattern to the DeepSeek provider but adapted for HuggingFace.

    Returns:
        modified_messages
    """
    # Get the JSON schema from the Pydantic model
    schema = pydantic_model.model_json_schema()

    # Add JSON instructions to the last user message
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""Answer the following question and format your response as a JSON object matching this schema:

Schema: {json.dumps(schema, indent=2)}

DO NOT return the schema itself. Instead, answer the question and put your answer in the correct JSON format.

For example, if the question asks for a name and you want to answer "Paris", return: {{"name": "Paris"}}

Question: {original_content}

Answer (as JSON):"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


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

        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to HuggingFace JSON format
                messages = _convert_pydantic_to_huggingface_json(response_format, messages)

        # Remove other unsupported parameters
        kwargs = remove_unsupported_params(kwargs, ["parallel_tool_calls"])

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
