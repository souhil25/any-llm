import os
from typing import Any

try:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model
except ImportError:
    msg = "mistralai is not installed. Please install it with `pip install any-llm-sdk[mistral]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
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


def _convert_response(response: Any) -> ChatCompletion:
    """Convert Mistral response directly to OpenAI ChatCompletion format."""
    # Convert Mistral response to dict
    response_dict = response.model_dump()

    # Build choices list
    choices = []
    for i, choice_data in enumerate(response_dict["choices"]):
        message_data = choice_data["message"]

        # Handle tool calls if present
        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"] is not None:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tool_call.get("id") or "unknown",
                    type="function",
                    function=Function(
                        name=tool_call.get("function", {}).get("name", ""),
                        arguments=tool_call.get("function", {}).get("arguments", ""),
                    ),
                )
                for tool_call in message_data["tool_calls"]
            ]

        # Create the message
        message = ChatCompletionMessage(
            content=message_data["content"],
            role=message_data.get("role", "assistant"),
            tool_calls=tool_calls,
        )

        # Create the choice
        choice = Choice(
            finish_reason=choice_data.get("finish_reason", "stop"),
            index=i,
            message=message,
        )
        choices.append(choice)

    # Create usage info if available
    usage = None
    if usage_data := response_dict.get("usage"):
        usage = CompletionUsage(
            completion_tokens=usage_data["completion_tokens"],
            prompt_tokens=usage_data["prompt_tokens"],
            total_tokens=usage_data["total_tokens"],
            prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            completion_tokens_details=usage_data.get("completion_tokens_details"),
        )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response_dict["id"],
        model=response_dict["model"],
        object="chat.completion",
        created=response_dict["created"],
        choices=choices,
        usage=usage,
    )


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

        return _convert_response(response)
