import os
import json
import urllib.request
import urllib.error
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert Azure response to OpenAI ChatCompletion format."""
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


class AzureProvider(Provider):
    """Azure Provider using urllib for direct API calls."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Azure provider."""
        self.base_url = config.api_base or os.getenv("AZURE_BASE_URL")
        self.api_key = config.api_key or os.getenv("AZURE_API_KEY")
        self.api_version = os.getenv("AZURE_API_VERSION")

        if not self.api_key:
            raise MissingApiKeyError("Azure", "AZURE_API_KEY")

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Azure."""
        if not self.base_url:
            raise ValueError(
                "For Azure, base_url is required. Check your deployment page for a URL like this - "
                "https://<model-deployment-name>.<region>.models.ai.azure.com"
            )

        # Build the URL
        url = f"{self.base_url}/chat/completions"
        if self.api_version:
            url = f"{url}?api-version={self.api_version}"

        # Prepare the request payload
        data = {"messages": messages}

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

        # Prepare the request
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key or "",
        }

        # Make the request to Azure endpoint
        req = urllib.request.Request(url, body, headers)
        with urllib.request.urlopen(req) as response:
            result = response.read()
            response_data = json.loads(result)

            # Convert to OpenAI format
            return _convert_response(response_data)
