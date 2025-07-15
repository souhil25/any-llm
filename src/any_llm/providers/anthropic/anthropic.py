import os
import json
from typing import Any

try:
    from anthropic import Anthropic
    from anthropic.types import Message
except ImportError:
    msg = "anthropic is not installed. Please install it with `pip install any-llm-sdk[anthropic]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Anthropic."""
    kwargs = kwargs.copy()
    kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

    # Convert tools if present
    if "tools" in kwargs:
        kwargs["tools"] = _convert_tool_spec(kwargs["tools"])

    # Handle parallel_tool_calls parameter
    if "parallel_tool_calls" in kwargs:
        parallel_tool_calls = kwargs.pop("parallel_tool_calls")
        # If parallel_tool_calls is False, set disable_parallel_tool_use to True
        if parallel_tool_calls is False:
            tool_choice = {"type": kwargs.get("tool_choice", "any"), "disable_parallel_tool_use": True}
            kwargs["tool_choice"] = tool_choice
        # If parallel_tool_calls is True or not specified, don't set disable_parallel_tool_use
        # (Anthropic defaults to parallel tool use enabled)

    if "response_format" in kwargs:
        error_msg = (
            "response_format is not supported for Anthropic, see their documentation "
            "for tips on how to achieve structured output: "
            "https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency#example-standardizing-customer-feedback"
        )
        raise ValueError(error_msg)

    return kwargs


def _convert_tool_spec(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool specification to Anthropic format."""
    anthropic_tools = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        anthropic_tool = {
            "name": function["name"],
            "description": function["description"],
            "input_schema": {
                "type": "object",
                "properties": function["parameters"]["properties"],
                "required": function["parameters"].get("required", []),
            },
        }
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert messages to Anthropic format, extracting system message."""
    system_message = ""
    converted_messages = []

    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
            continue
        elif message["role"] == "tool":
            # Convert tool message to Anthropic format
            converted_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                ],
            }
            converted_messages.append(converted_message)
        elif message["role"] == "assistant" and "tool_calls" in message:
            # Convert assistant message with tool calls
            message_content = []
            if message.get("content"):
                message_content.append({"type": "text", "text": message["content"]})

            for tool_call in message.get("tool_calls") or []:
                message_content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"]),
                    }
                )

            converted_message = {"role": "assistant", "content": message_content}
            converted_messages.append(converted_message)
        else:
            # Regular message
            converted_message = {"role": message["role"], "content": message["content"]}
            converted_messages.append(converted_message)

    return system_message, converted_messages


def _convert_response(response: Message) -> ChatCompletion:
    """Convert Anthropic response directly to OpenAI ChatCompletion format."""
    # Finish reason mapping
    finish_reason_mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }

    # Process content blocks
    tool_calls = []
    content = ""

    for content_block in response.content:
        if content_block.type == "text":
            content = content_block.text
        elif content_block.type == "tool_use":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=content_block.id,
                    type="function",
                    function=Function(name=content_block.name, arguments=json.dumps(content_block.input)),
                )
            )

    # Create the message
    message = ChatCompletionMessage(
        content=content or None,
        role="assistant",
        tool_calls=tool_calls if tool_calls else None,
    )

    # Create the choice
    if not response.stop_reason:
        response.stop_reason = "end_turn"
    mapped_finish_reason = finish_reason_mapping.get(response.stop_reason, "stop")
    choice = Choice(
        finish_reason=mapped_finish_reason,  # type: ignore
        index=0,
        message=message,
    )

    # Create usage information
    usage = CompletionUsage(
        completion_tokens=response.usage.output_tokens,
        prompt_tokens=response.usage.input_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response.id,
        model=response.model,
        object="chat.completion",
        created=int(response.created_at.timestamp()) if hasattr(response, "created_at") else 0,
        choices=[choice],
        usage=usage,
    )


class AnthropicProvider(Provider):
    def __init__(self, config: ApiConfig) -> None:
        """Initialize Anthropic provider."""
        if not config.api_key:
            config.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not config.api_key:
            msg = "No Anthropic API key provided. Please provide it in the config or set the ANTHROPIC_API_KEY environment variable."
            raise ValueError(msg)
        self.client = Anthropic(api_key=config.api_key, base_url=config.api_base)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Anthropic."""
        kwargs = _convert_kwargs(kwargs)
        system_message, converted_messages = _convert_messages(messages)

        # Make the request to Anthropic
        response = self.client.messages.create(
            model=model,
            system=system_message,
            messages=converted_messages,  # type: ignore
            **kwargs,
        )

        # Convert to OpenAI format
        return _convert_response(response)
