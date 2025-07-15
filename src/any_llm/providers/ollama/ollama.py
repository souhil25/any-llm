import os
from datetime import datetime
from typing import Any
import uuid
import json

import httpx
from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.logging import logger
from any_llm.provider import ApiConfig, Provider


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert kwargs to Ollama format.
    """
    # Ensure stream is disabled for synchronous response
    kwargs["stream"] = False
    if "response_format" in kwargs:
        response_format = kwargs.pop("response_format")
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # response_format is a Pydantic model class
            kwargs["format"] = response_format.model_json_schema()
        else:
            # response_format is already a dict/schema
            kwargs["format"] = response_format
    return kwargs


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert messages to Ollama format.
    """
    for message in messages:
        if message["role"] == "tool":
            message["role"] = "user"
            message["content"] = json.dumps(message["content"])
            message.pop("tool_call_id")
            message.pop("name")
        elif message["role"] == "assistant" and "tool_calls" in message:
            message["content"] = message["content"] + "\n" + json.dumps(message["tool_calls"])
            message.pop("tool_calls")
    return messages


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """
    Convert Ollama response directly to OpenAI ChatCompletion format.
    """
    # Ollama returns a different format than OpenAI, so we need to transform it
    # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "model": "...", ...}
    created_str = response_data.get("created_at", 0)  # 2025-07-14T12:00:00Z
    # convert the string timestamp to int (2025-07-14T12:00:00Z)
    created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

    # Handle tool calls if present
    tool_calls = []
    if "tool_calls" in response_data["message"]:
        for tool_call in response_data["message"]["tool_calls"]:
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=str(uuid.uuid4()),
                    type="function",
                    function=Function(
                        name=tool_call["function"]["name"],
                        arguments=json.dumps(tool_call["function"]["arguments"]),
                    ),
                )
            )

    # Create the message
    message = ChatCompletionMessage(
        role=response_data["message"]["role"],
        content=response_data["message"]["content"],
        tool_calls=tool_calls if tool_calls else None,
    )

    # Create the choice
    choice = Choice(
        index=0,
        message=message,
        finish_reason=response_data["done_reason"],
    )

    # Create usage info if available
    usage = None
    if "prompt_eval_count" in response_data or "eval_count" in response_data:
        usage = CompletionUsage(
            prompt_tokens=response_data.get("prompt_eval_count", 0),
            completion_tokens=response_data.get("eval_count", 0),
            total_tokens=response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0),
        )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id="chatcmpl-" + str(hash(str(response_data))),  # Generate a fake ID
        object="chat.completion",
        created=created,
        model=response_data["model"],
        choices=[choice],
        usage=usage,
    )


class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls to the Ollama API.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _DEFAULT_URL = "http://localhost:11434"
    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Ollama provider."""
        self.url = str(config.api_base or os.getenv("OLLAMA_API_URL", self._DEFAULT_URL))

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Ollama."""
        kwargs = _convert_kwargs(kwargs)

        # Convert tool messages to user messages and remove tool_calls from assistant messages
        # (https://www.reddit.com/r/ollama/comments/1ked8x2/feeding_tool_output_back_to_llm/)
        messages = _convert_messages(messages)

        data = {
            "model": model,
            "messages": messages,
            "options": {"num_ctx": 32000},  # Default is 4096 which is too small for most use cases.
            **kwargs,  # Pass any additional arguments to the API
        }
        timeout = int(kwargs.pop("timeout", 30))
        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Error calling Ollama: {e.response.text}")
            raise e
        response_data = response.json()
        # Convert Ollama response to OpenAI format
        return _convert_response(response_data)
