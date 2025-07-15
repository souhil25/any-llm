import os
from typing import Any

try:
    import groq
except ImportError:
    msg = "groq is not installed. Please install it with `pip install any-llm-sdk[groq]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Groq."""
    # Since Groq is OpenAI-compliant, we can pass most kwargs through
    kwargs = kwargs.copy()
    
    # Handle any unsupported parameters if needed
    return kwargs


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to Groq format."""
    # Since Groq is OpenAI-compliant, minimal conversion needed
    converted_messages = []
    
    for message in messages:
        # Remove refusal field if present (following aisuite pattern)
        converted_message = message.copy()
        converted_message.pop("refusal", None)
        
        converted_messages.append(converted_message)
    
    return converted_messages


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert Groq response to OpenAI ChatCompletion format."""
    # Since Groq is OpenAI-compliant, the response should already be in the right format
    # We just need to create proper OpenAI objects
    
    choice_data = response_data["choices"][0]
    message_data = choice_data["message"]
    
    # Handle tool calls if present
    tool_calls = None
    if "tool_calls" in message_data and message_data["tool_calls"]:
        tool_calls = []
        for tool_call in message_data["tool_calls"]:
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call.get("id"),
                    type="function",  # Always set to "function" as it's the only valid value
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
        finish_reason=choice_data.get("finish_reason", "stop"),  # type: ignore
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


class GroqProvider(Provider):
    """Groq Provider using the official Groq client."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Groq provider."""
        if not config.api_key:
            config.api_key = os.getenv("GROQ_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError(
                "Groq",
                "GROQ_API_KEY",
            )
        
        # Initialize the Groq client
        client_config = {"api_key": config.api_key}
        if config.api_base:
            client_config["base_url"] = config.api_base
        
        self.client = groq.Groq(**client_config)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Groq."""
        kwargs = _convert_kwargs(kwargs)
        converted_messages = _convert_messages(messages)
        
        try:
            # Make the API call using the client
            response = self.client.chat.completions.create(
                model=model,
                messages=converted_messages,
                **kwargs,
            )
            
            # Convert response to dict format for processing
            response_data = response.model_dump()
            
            # Convert to OpenAI format
            return _convert_response(response_data)
            
        except Exception as e:
            # Re-raise as a more generic exception
            raise RuntimeError(f"Groq API error: {e}") from e 