import os
import json
from typing import Any

try:
    import cerebras.cloud.sdk as cerebras
except ImportError:
    msg = "cerebras is not installed. Please install it with `pip install any-llm-sdk[cerebras]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Cerebras."""
    # Since Cerebras is OpenAI-compliant, we can pass most kwargs through
    kwargs = kwargs.copy()

    # Handle any unsupported parameters if needed
    return kwargs


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to Cerebras format."""
    # Since Cerebras is OpenAI-compliant, minimal conversion needed
    converted_messages = []

    for message in messages:
        # Remove refusal field if present (following aisuite pattern)
        converted_message = message.copy()
        converted_message.pop("refusal", None)

        # Handle tool messages - convert content to string if needed
        if message.get("role") == "tool":
            if "content" in converted_message:
                converted_message["content"] = str(converted_message["content"])

        converted_messages.append(converted_message)

    return converted_messages


def _add_json_instruction_to_messages(messages: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Add JSON instruction to the last user message for structured output."""
    if not messages or messages[-1]["role"] != "user":
        return messages

    # Create a copy of messages to avoid modifying the original
    modified_messages = messages.copy()
    original_content = modified_messages[-1]["content"]

    json_instruction = f"""
Please respond with a JSON object that matches the following schema:

{json.dumps(schema, indent=2)}

Return the JSON object only, no other text.

{original_content}
"""
    modified_messages[-1]["content"] = json_instruction

    return modified_messages


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert Cerebras response to OpenAI ChatCompletion format."""
    # Since Cerebras is OpenAI-compliant, the response should already be in the right format
    # We just need to create proper OpenAI objects

    choice_data = response_data["choices"][0]
    message_data = choice_data["message"]

    # Handle tool calls if present
    tool_calls = None
    if "tool_calls" in message_data and message_data["tool_calls"]:
        tool_calls = []
        for tool_call in message_data["tool_calls"]:
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call.get("id"),
                    type="function",  # Always set to "function" as it's the only valid value
                    function=Function(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    ),
                )
            )

    # Create the message
    message = ChatCompletionMessage(
        content=message_data.get("content"),
        role=message_data.get("role", "assistant"),
        tool_calls=tool_calls,
    )

    # Create the choice
    choice = Choice(
        finish_reason=choice_data.get("finish_reason", "stop"),
        index=choice_data.get("index", 0),
        message=message,
    )

    # Create usage information (if available)
    usage = None
    if "usage" in response_data:
        usage_data = response_data["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response_data.get("id", ""),
        model=response_data.get("model", ""),
        object="chat.completion",
        created=response_data.get("created", 0),
        choices=[choice],
        usage=usage,
    )


class CerebrasProvider(Provider):
    """Cerebras Provider using the official Cerebras SDK."""

    PROVIDER_NAME = "Cerebras"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cerebras provider."""
        if not config.api_key:
            config.api_key = os.getenv("CEREBRAS_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError(
                "Cerebras",
                "CEREBRAS_API_KEY",
            )

        # Initialize the Cerebras client
        self.client = cerebras.Cerebras(api_key=config.api_key)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Cerebras."""
        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to JSON schema format for Cerebras
                schema = response_format.model_json_schema()
                kwargs["response_format"] = {"type": "json_object"}

                # Add JSON instruction to the last user message (required by Cerebras)
                messages = _add_json_instruction_to_messages(messages, schema)

        kwargs = _convert_kwargs(kwargs)
        converted_messages = _convert_messages(messages)

        # Make the API call using the client
        response = self.client.chat.completions.create(
            model=model,
            messages=converted_messages,
            **kwargs,
        )

        # Convert response to dict format for processing
        # Handle the case where response might be a Stream object
        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        else:
            # If it's a streaming response, we need to handle it differently
            raise ValueError("Streaming responses are not supported in this context")

        # Convert to OpenAI format
        return _convert_response(response_data)
