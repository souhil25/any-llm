from typing import Any

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall


def convert_response_to_openai(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert response to OpenAI format."""
    # Build choices list first
    choices = []
    for i, choice_data in enumerate(response_data["choices"]):
        message_data = choice_data["message"]

        # Handle tool calls if present
        tool_calls = None
        if "tool_calls" in message_data and message_data["tool_calls"] is not None:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tool_call.get("id"),
                    type="function",  # Always set to "function" as it's the only valid value
                    function=tool_call.get("function"),
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

    # Create the completion response with populated choices
    completion_response = ChatCompletion(
        id=response_data["id"],
        model=response_data["model"],
        object="chat.completion",
        created=response_data["created"],
        choices=choices,
    )

    # Conditionally parse usage data if it exists.
    if usage_data := response_data.get("usage"):
        completion_response.usage = convert_usage_to_openai(usage_data)

    return completion_response


def convert_usage_to_openai(usage_data: dict[str, Any]) -> CompletionUsage:
    """Get the completion usage."""
    return CompletionUsage(
        completion_tokens=usage_data["completion_tokens"],
        prompt_tokens=usage_data["prompt_tokens"],
        total_tokens=usage_data["total_tokens"],
        prompt_tokens_details=usage_data.get("prompt_tokens_details"),
        completion_tokens_details=usage_data.get("completion_tokens_details"),
    )
