import os
from typing import Any

try:
    import httpx
except ImportError:
    msg = "httpx is not installed. Please install it with `pip install any-llm-sdk[fireworks]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Fireworks."""
    kwargs = kwargs.copy()

    # Remove 'stream' from kwargs if present (not supported in this implementation)
    kwargs.pop("stream", None)

    return kwargs


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to Fireworks format."""
    converted_messages = []

    for message in messages:
        # Remove refusal field if present (following aisuite pattern)
        converted_message = message.copy()
        converted_message.pop("refusal", None)

        converted_messages.append(converted_message)

    return converted_messages


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert Fireworks response to OpenAI ChatCompletion format."""
    choice_data = response_data["choices"][0]
    message_data = choice_data["message"]

    # Handle tool calls if present
    tool_calls = None
    if "tool_calls" in message_data and message_data["tool_calls"]:
        tool_calls = []
        for tool_call in message_data["tool_calls"]:
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call["id"],
                    type=tool_call["type"],
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


class FireworksProvider(Provider):
    """Fireworks AI Provider using httpx for direct API calls."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Fireworks provider."""
        if not config.api_key:
            config.api_key = os.getenv("FIREWORKS_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError(
                "Fireworks",
                "FIREWORKS_API_KEY",
            )

        self.api_key = config.api_key
        self.base_url = config.api_base or BASE_URL
        self.timeout = 30  # Default timeout

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Fireworks."""
        kwargs = _convert_kwargs(kwargs)
        converted_messages = _convert_messages(messages)

        # Prepare the request payload
        data = {
            "model": model,
            "messages": converted_messages,
        }

        # Add tools if provided
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
            kwargs.pop("tools")

        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs["tool_choice"]
            kwargs.pop("tool_choice")

        # Add remaining kwargs
        data.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            # Make the request to Fireworks AI endpoint
            response = httpx.post(
                self.base_url,
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Convert to OpenAI format
            response_data = response.json()
            return _convert_response(response_data)

        except httpx.HTTPStatusError as error:
            error_message = (
                f"The request failed with status code: {error.response.status_code}\n"
                f"Headers: {error.response.headers}\n"
                f"{error.response.text}"
            )
            raise RuntimeError(f"Fireworks API error: {error_message}") from error
        except Exception as e:
            # Re-raise as a more generic exception
            raise RuntimeError(f"Fireworks API error: {e}") from e
