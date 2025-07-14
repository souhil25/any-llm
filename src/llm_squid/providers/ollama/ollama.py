import os
from datetime import datetime
from typing import Any
import uuid
import json

import httpx
from openai.types.chat.chat_completion import ChatCompletion

from llm_squid.utils import convert_response_to_openai
from llm_squid.utils.provider import Provider


class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls to the Ollama API.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, **config: Any) -> None:
        """Initialize Ollama provider."""
        self.url = config.get("api_url") or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Ollama."""
        # Ensure stream is disabled for synchronous response
        kwargs["stream"] = False
        if "response_format" in kwargs:
            kwargs["format"] = kwargs.pop("response_format").model_json_schema()

        # Convert tool messages to user messages
        for message in messages:
            if message["role"] == "tool":
                message["role"] = "user"
                message["content"] = json.dumps(message["content"])
                message.pop("tool_call_id")
                message.pop("name")
            elif message["role"] == "assistant":
                message.pop("tool_calls", None)

        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        response = httpx.post(
            self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_data = response.json()
        # Convert Ollama response to OpenAI format
        return self._normalize_response(response_data)

    def _normalize_response(self, response_data: dict[str, Any]) -> ChatCompletion:
        """
        Convert Ollama response to OpenAI format.
        """
        # Ollama returns a different format than OpenAI, so we need to transform it
        # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "model": "...", ...}
        created_str = response_data.get("created_at", 0)  # 2025-07-14T12:00:00Z
        # convert the string timestamp to int (2025-07-14T12:00:00Z)
        created = int(datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

        tool_calls = []
        if "tool_calls" in response_data["message"]:
            for tool_call in response_data["message"]["tool_calls"]:
                tool_calls.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(tool_call["function"]["arguments"]),
                        },
                    }
                )

        # Transform to OpenAI format
        openai_format = {
            "id": "chatcmpl-" + str(hash(str(response_data))),  # Generate a fake ID
            "object": "chat.completion",
            "created": created,
            "model": response_data["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": response_data["message"]["role"],
                        "content": response_data["message"]["content"],
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": response_data["done_reason"],
                }
            ],
        }

        # Add usage info if available
        if "prompt_eval_count" in response_data or "eval_count" in response_data:
            openai_format["usage"] = {
                "prompt_tokens": response_data.get("prompt_eval_count", 0),
                "completion_tokens": response_data.get("eval_count", 0),
                "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0),
            }

        return convert_response_to_openai(openai_format)
