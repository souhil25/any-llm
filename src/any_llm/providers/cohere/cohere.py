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
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.base_framework import (
    create_completion_from_response,
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

    PROVIDER_NAME = "Cohere"

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
        if "response_format" in kwargs:
            raise UnsupportedParameterError("response_format", self.PROVIDER_NAME)
        elif "parallel_tool_calls" in kwargs:
            raise UnsupportedParameterError("parallel_tool_calls", self.PROVIDER_NAME)

        # Make the API call
        response = self.client.chat(
            model=model,
            messages=messages,  # type: ignore[arg-type]
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
