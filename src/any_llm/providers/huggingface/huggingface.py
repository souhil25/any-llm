import os
import json
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    msg = "huggingface-hub is not installed. Please install it with `pip install any-llm-sdk[huggingface]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for HuggingFace."""
    kwargs = kwargs.copy()
    
    # HuggingFace typically uses max_new_tokens instead of max_tokens
    if "max_tokens" in kwargs:
        kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
    
    # Handle unsupported parameters
    unsupported_params = ["response_format", "parallel_tool_calls"]
    for param in unsupported_params:
        if param in kwargs:
            kwargs.pop(param)
    
    return kwargs


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to HuggingFace format."""
    converted_messages = []
    
    for message in messages:
        # Ensure content is a string
        content = message.get("content", "")
        if content is None:
            content = ""
        
        converted_message = {
            "role": message["role"],
            "content": content,
        }
        
        # Handle tool calls if present
        if "tool_calls" in message and message["tool_calls"]:
            converted_message["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                    },
                    "type": tool_call["type"],
                }
                for tool_call in message["tool_calls"]
            ]
        
        # Handle tool call ID for tool messages
        if "tool_call_id" in message:
            converted_message["tool_call_id"] = message["tool_call_id"]
        
        converted_messages.append(converted_message)
    
    return converted_messages


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert HuggingFace response to OpenAI ChatCompletion format."""
    choice_data = response["choices"][0]
    message_data = choice_data["message"]
    
    # Handle tool calls in the response
    tool_calls = None
    if "tool_calls" in message_data and message_data["tool_calls"]:
        tool_calls = []
        for tool_call in message_data["tool_calls"]:
            # Ensure function arguments are stringified
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call["id"],
                    type="function",
                    function=Function(
                        name=tool_call["function"]["name"],
                        arguments=arguments,
                    ),
                )
            )
    
    # Create the message
    message = ChatCompletionMessage(
        content=message_data.get("content", ""),
        role="assistant",
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
    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("completion_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
    
    # Build the final ChatCompletion object
    return ChatCompletion(
        id=response.get("id", ""),
        model=response.get("model", ""),
        object="chat.completion",
        created=response.get("created", 0),
        choices=[choice],
        usage=usage,
    )


class HuggingfaceProvider(Provider):
    """HuggingFace Provider using the official InferenceClient."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize HuggingFace provider."""
        if not config.api_key:
            config.api_key = os.getenv("HF_TOKEN")
        if not config.api_key:
            raise MissingApiKeyError(
                "HuggingFace",
                "HF_TOKEN",
            )
        
        # Initialize the InferenceClient
        self.client = InferenceClient(
            token=config.api_key,
            timeout=30  # Default timeout
        )

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using HuggingFace."""
        kwargs = _convert_kwargs(kwargs)
        converted_messages = _convert_messages(messages)
        
        try:
            # Make the API call using the client
            response = self.client.chat_completion(
                model=model,
                messages=converted_messages,
                **kwargs,
            )
            
            # Convert to OpenAI format
            return _convert_response(response)
            
        except Exception as e:
            # Re-raise as a more generic exception
            raise RuntimeError(f"HuggingFace API error: {e}") from e 