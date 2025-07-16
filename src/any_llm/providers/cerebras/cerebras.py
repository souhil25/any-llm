import os
from typing import Any, cast

try:
    import cerebras.cloud.sdk as cerebras
    import instructor
except ImportError:
    msg = "cerebras or instructor is not installed. Please install it with `pip install any-llm-sdk[cerebras]`"
    raise ImportError(msg)


from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.exceptions import MissingApiKeyError


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Cerebras."""
    # Since Cerebras is OpenAI-compliant, we can pass most kwargs through
    kwargs = kwargs.copy()

    # Remove response_format since it will be handled by instructor
    kwargs.pop("response_format", None)

    return kwargs


def _convert_response(response_data: dict[str, Any]) -> ChatCompletion:
    """Convert Cerebras response to OpenAI ChatCompletion format."""
    # Since Cerebras is OpenAI-compliant, the response should already be in the right format
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


class CerebrasProvider(Provider):
    """Cerebras Provider using the official Cerebras SDK with instructor support for structured outputs."""

    PROVIDER_NAME = "Cerebras"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cerebras provider."""
        if not config.api_key:
            config.api_key = os.getenv("CEREBRAS_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError(
                "Cerebras",
                "CEREBRAS_API_KEY",
            )

        # Initialize the Cerebras client
        self.client = cerebras.Cerebras(api_key=config.api_key)

        # Create instructor client for structured output support
        self.instructor_client = instructor.from_cerebras(self.client)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Cerebras with instructor support for structured outputs."""

        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            converted_kwargs = _convert_kwargs(kwargs)

            # Use instructor for structured output
            instructor_response = self.instructor_client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                response_model=response_format,
                **converted_kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "cerebras")

        # For non-structured outputs, use the regular client
        kwargs = _convert_kwargs(kwargs)

        # Use regular create method for non-structured outputs
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        # Convert response to dict format for processing
        # Handle the case where response might be a Stream object
        if hasattr(response, "model_dump"):
            response_data = response.model_dump()
        else:
            # If it's a streaming response, we need to handle it differently
            raise ValueError("Streaming responses are not supported in this context")

        # Convert to OpenAI format
        return _convert_response(response_data)
