from typing import Any

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall


class OpenAICompliantMessageConverter:
    """Convert messages to OpenAI format."""

    tool_results_as_strings = False

    @staticmethod
    def convert_request(
        messages: list[ChatCompletionMessage | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format."""
        transformed_messages = []
        for message in messages:
            tmsg = None
            if isinstance(message, ChatCompletionMessage):
                message_dict = message.model_dump(mode="json")
                message_dict.pop("refusal", None)  # Remove refusal field if present
                tmsg = message_dict
            else:
                tmsg = message
            # Check if tmsg is a dict, otherwise get role attribute
            if tmsg is not None:
                role = tmsg["role"] if isinstance(tmsg, dict) else tmsg.role
                if role == "tool" and OpenAICompliantMessageConverter.tool_results_as_strings:
                    # Handle both dict and object cases for content
                    if isinstance(tmsg, dict):
                        tmsg["content"] = str(tmsg["content"])
                    else:
                        tmsg.content = str(tmsg.content)
                transformed_messages.append(tmsg)
        return transformed_messages

    def convert_response(self, response_data: dict[str, Any]) -> ChatCompletion:
        """Convert response to OpenAI format."""
        # Build choices list first
        choices = []
        for i, choice_data in enumerate(response_data["choices"]):
            message_data = choice_data["message"]

            # Handle tool calls if present
            tool_calls = None
            if "tool_calls" in message_data and message_data["tool_calls"] is not None:
                tool_calls = [
                    ChatCompletionMessageToolCall(  # type: ignore[reportUnknownMemberType]
                        id=tool_call.get("id"),
                        type="function",  # Always set to "function" as it's the only valid value
                        function=tool_call.get("function"),
                    )
                    for tool_call in message_data["tool_calls"]
                ]

            # Create the message
            message = ChatCompletionMessage(
                content=message_data["content"],
                role=message_data.get("role", "assistant"),
                tool_calls=tool_calls,
            )

            # Create the choice
            choice = Choice(
                finish_reason=choice_data.get("finish_reason", "stop"),
                index=i,
                message=message,
            )
            choices.append(choice)

        # Create the completion response with populated choices
        completion_response = ChatCompletion(
            id=response_data["id"],
            model=response_data["model"],
            object="chat.completion",
            created=response_data["created"],
            choices=choices,
        )

        # Conditionally parse usage data if it exists.
        if usage_data := response_data.get("usage"):
            completion_response.usage = self.get_completion_usage(usage_data)

        return completion_response

    def get_completion_usage(self, usage_data: dict[str, Any]) -> CompletionUsage:
        """Get the completion usage."""
        return CompletionUsage(
            completion_tokens=usage_data["completion_tokens"],
            prompt_tokens=usage_data["prompt_tokens"],
            total_tokens=usage_data["total_tokens"],
            prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            completion_tokens_details=usage_data.get("completion_tokens_details"),
        )
