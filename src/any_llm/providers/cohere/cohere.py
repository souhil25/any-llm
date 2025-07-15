import os
import json
from typing import Any

try:
    import cohere
except ImportError:
    msg = "cohere is not installed. Please install it with `pip install any-llm-sdk[cohere]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Cohere."""
    kwargs = kwargs.copy()
    
    # Handle unsupported parameters
    unsupported_params = ["response_format", "parallel_tool_calls"]
    for param in unsupported_params:
        if param in kwargs:
            kwargs.pop(param)
    
    return kwargs


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


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages to Cohere format."""
    converted_messages = []
    
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        tool_plan = message.get("tool_plan")
        
        # Convert to Cohere's format
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
    
    return converted_messages


def _convert_response(response: Any) -> ChatCompletion:
    """Convert Cohere response to OpenAI ChatCompletion format."""
    # Create usage information
    usage = CompletionUsage(
        prompt_tokens=response.usage.tokens.input_tokens,
        completion_tokens=response.usage.tokens.output_tokens,
        total_tokens=response.usage.tokens.input_tokens + response.usage.tokens.output_tokens,
    )
    
    # Handle tool calls
    if response.finish_reason == "TOOL_CALL":
        tool_call = response.message.tool_calls[0]
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=tool_call.id,
                type="function",
                function=Function(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
        ]
        
        # Create the message
        message = ChatCompletionMessage(
            content=response.message.tool_plan,  # Use tool_plan as content
            role="assistant",
            tool_calls=tool_calls,
        )
        
        finish_reason = "tool_calls"
    else:
        # Handle regular text response
        message = ChatCompletionMessage(
            content=response.message.content[0].text,
            role="assistant",
            tool_calls=None,
        )
        
        finish_reason = "stop"
    
    # Create the choice
    choice = Choice(
        finish_reason=finish_reason,  # type: ignore
        index=0,
        message=message,
    )
    
    # Build the final ChatCompletion object
    return ChatCompletion(
        id=getattr(response, "id", ""),
        model=getattr(response, "model", ""),
        object="chat.completion",
        created=getattr(response, "created", 0),
        choices=[choice],
        usage=usage,
    )


class CohereProvider(Provider):
    """Cohere Provider using the official Cohere client."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cohere provider."""
        if not config.api_key:
            config.api_key = os.getenv("CO_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError(
                "Cohere",
                "CO_API_KEY",
            )
        
        # Initialize the Cohere client
        client_config = {"api_key": config.api_key}
        if config.api_base:
            client_config["base_url"] = config.api_base
        
        self.client = cohere.ClientV2(**client_config)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Cohere."""
        kwargs = _convert_kwargs(kwargs)
        converted_messages = _convert_messages(messages)
        
        try:
            # Make the API call using the client
            response = self.client.chat(
                model=model,
                messages=converted_messages,
                **kwargs,
            )
            
            # Convert to OpenAI format
            return _convert_response(response)
            
        except Exception as e:
            # Re-raise as a more generic exception
            raise RuntimeError(f"Cohere API error: {e}") from e 