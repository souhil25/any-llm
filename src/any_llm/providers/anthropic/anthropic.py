import json
from typing import Any, cast

from any_llm.logging import logger

try:
    from anthropic import Anthropic
    from anthropic.types import Message
    import instructor
except ImportError:
    msg = "anthropic or instructor is not installed. Please install it with `pip install any-llm-sdk[anthropic]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage

from any_llm.provider import ApiConfig, convert_instructor_response
from any_llm.providers.base_framework import (
    BaseCustomProvider,
    create_openai_tool_call,
    create_openai_message,
    create_openai_completion,
    convert_openai_tools_to_generic,
    extract_system_message,
)

DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(BaseCustomProvider):
    """
    Anthropic Provider using enhanced BaseCustomProvider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "Anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"

    def _initialize_client(self, config: ApiConfig) -> None:
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=config.api_key, base_url=config.api_base)

        # Create instructor client for structured output
        self.instructor_client = instructor.from_anthropic(self.client)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Anthropic with instructor support."""
        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")

            # Convert messages to Anthropic format
            system_message, converted_messages = self._convert_messages(messages)

            # Convert other kwargs
            converted_kwargs = self._convert_kwargs(kwargs)

            # Use instructor for structured output
            instructor_response = self.instructor_client.messages.create(
                model=model,
                system=system_message,
                messages=converted_messages,
                response_model=response_format,
                **converted_kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "anthropic")

        # Fall back to standard completion flow
        return super().completion(model, messages, **kwargs)

    def _convert_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert kwargs to Anthropic format."""
        kwargs = kwargs.copy()

        if "max_tokens" not in kwargs:
            logger.warning(f"max_tokens is required for Anthropic, setting to {DEFAULT_MAX_TOKENS}")
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # Convert tools if present
        if "tools" in kwargs:
            kwargs["tools"] = self._convert_tool_spec(kwargs["tools"])

        # Handle parallel_tool_calls
        if "parallel_tool_calls" in kwargs:
            parallel_tool_calls = kwargs.pop("parallel_tool_calls")
            if parallel_tool_calls is False:
                tool_choice = {"type": kwargs.get("tool_choice", "any"), "disable_parallel_tool_use": True}
                kwargs["tool_choice"] = tool_choice

        return kwargs

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Convert messages to Anthropic format, extracting system message."""
        # Extract system message using the utility
        system_message, remaining_messages = extract_system_message(messages)

        converted_messages = []
        for message in remaining_messages:
            if message["role"] == "tool":
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
                converted_message = {"role": message["role"], "content": message["content"]}
                converted_messages.append(converted_message)

        return system_message, converted_messages

    def _make_api_call(self, model: str, messages: tuple[str, list[dict[str, Any]]], **kwargs: Any) -> Message:
        """Make the API call to Anthropic."""
        system_message, converted_messages = messages

        return self.client.messages.create(
            model=model,
            system=system_message,
            messages=converted_messages,  # type: ignore[arg-type]
            **kwargs,
        )

    def _convert_response(self, response: Message) -> ChatCompletion:
        """Convert Anthropic response to OpenAI format."""
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
                    create_openai_tool_call(
                        tool_call_id=content_block.id,
                        name=content_block.name,
                        arguments=json.dumps(content_block.input),
                    )
                )

        # Create the message
        message = create_openai_message(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Create the choice
        mapped_finish_reason = finish_reason_mapping.get(response.stop_reason or "end_turn", "stop")
        choice = Choice(
            finish_reason=cast(Any, mapped_finish_reason),
            index=0,
            message=message,
        )

        # Create usage information
        usage = CompletionUsage(
            completion_tokens=response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return create_openai_completion(
            id=response.id,
            model=response.model,
            choices=[choice],
            usage=usage,
            created=int(response.created_at.timestamp()) if hasattr(response, "created_at") else 0,
        )

    def _convert_tool_spec(self, openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool specification to Anthropic format."""
        # Use the generic utility first
        generic_tools = convert_openai_tools_to_generic(openai_tools)

        # Convert to Anthropic-specific format
        anthropic_tools = []
        for tool in generic_tools:
            anthropic_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["parameters"]["properties"],
                    "required": tool["parameters"].get("required", []),
                },
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools
