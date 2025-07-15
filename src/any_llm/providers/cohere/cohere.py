import os
import json
from typing import Any

try:
    import cohere
except ImportError:
    msg = "cohere is not installed. Please install it with `pip install any-llm-sdk[cohere]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.base_framework import (
    create_completion_from_response,
    remove_unsupported_params,
)


def _convert_tool_content(content: Any) -> Any:
    """Convert tool response content to Cohere's expected format."""
    if isinstance(content, str):
        try:
            # Try to parse as JSON first
            data = json.loads(content)
            return [{"type": "document", "document": {"data": json.dumps(data)}}]
        except json.JSONDecodeError:
            # If not JSON, return as plain text
            return content
    elif isinstance(content, list):
        # If content is already in Cohere's format, return as is
        return content
    else:
        # For other types, convert to string
        return str(content)


class CohereProvider(Provider):
    """Cohere Provider using the new response conversion utilities."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cohere provider."""
        if not config.api_key:
            config.api_key = os.getenv("CO_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Cohere", "CO_API_KEY")

        self.client = cohere.ClientV2(api_key=config.api_key)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Cohere."""
        # Remove unsupported parameters
        kwargs = remove_unsupported_params(kwargs, ["response_format", "parallel_tool_calls"])

        # Convert messages to Cohere format
        converted_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            tool_plan = message.get("tool_plan")

            if role == "tool":
                # Handle tool response messages
                converted_message = {
                    "role": role,
                    "tool_call_id": message.get("tool_call_id"),
                    "content": _convert_tool_content(content),
                }
            elif role == "assistant" and tool_calls:
                # Handle assistant messages with tool calls
                converted_message = {
                    "role": role,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                            "type": "function",
                        }
                        for tc in tool_calls
                    ],
                    "tool_plan": tool_plan,
                }
                if content:
                    converted_message["content"] = content
            else:
                # Handle regular messages
                converted_message = {"role": role, "content": content}

            converted_messages.append(converted_message)

        try:
            # Make the API call
            response = self.client.chat(
                model=model,
                messages=converted_messages,  # type: ignore[arg-type]
                **kwargs,
            )

            # Convert response to dict-like structure for the utility
            prompt_tokens = 0
            completion_tokens = 0

            if response.usage and response.usage.tokens:
                prompt_tokens = int(response.usage.tokens.input_tokens or 0)
                completion_tokens = int(response.usage.tokens.output_tokens or 0)

            response_dict = {
                "id": getattr(response, "id", ""),
                "model": getattr(response, "model", ""),
                "created": getattr(response, "created", 0),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

            # Handle tool calls vs regular responses
            if response.finish_reason == "TOOL_CALL" and response.message.tool_calls:
                tool_call = response.message.tool_calls[0]
                response_dict["choices"] = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response.message.tool_plan,  # Use tool_plan as content
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_call.function.name if tool_call.function else "",
                                        "arguments": tool_call.function.arguments if tool_call.function else "",
                                    },
                                    "type": "function",
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "index": 0,
                    }
                ]
            else:
                # Regular text response
                content = ""
                if response.message.content and len(response.message.content) > 0:
                    content = response.message.content[0].text

                response_dict["choices"] = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ]

            # Convert to OpenAI format using the new utility
            return create_completion_from_response(
                response_data=response_dict,
                model=model,
                provider_name="cohere",
            )

        except Exception as e:
            raise RuntimeError(f"Cohere API error: {e}") from e
