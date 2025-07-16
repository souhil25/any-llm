import json
from typing import Any

from pydantic import BaseModel

from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion


def _convert_pydantic_to_deepseek_json(
    pydantic_model: type[BaseModel], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert Pydantic model to DeepSeek JSON format.

    DeepSeek requires:
    1. response_format = {'type': 'json_object'}
    2. The word "json" in the prompt
    3. An example of the desired JSON structure

    Following the instructions in the DeepSeek documentation:
    https://api-docs.deepseek.com/guides/json_mode

    Returns:
        modified_messages
    """
    # Get the JSON schema from the Pydantic model
    schema = pydantic_model.model_json_schema()

    # Add JSON instructions to the last user message
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""
Please respond with a JSON object that can be loaded into a pydantic model that matches the following schema:

{json.dumps(schema, indent=2)}

Return the JSON object only, no other text, do not wrap it in ```json or ```.

{original_content}
"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages


class DeepseekProvider(BaseOpenAIProvider):
    """
    DeepSeek Provider implementation.

    This provider connects to DeepSeek's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use DeepSeek's configuration.

    Configuration:
    - api_key: DeepSeek API key (can be set via DEEPSEEK_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to DeepSeek's API)

    Example usage:
        config = ApiConfig(api_key="your-deepseek-api-key")
        provider = DeepseekProvider(config)
        response = provider.completion("deepseek-chat", messages=[...])
    """

    # DeepSeek-specific configuration
    DEFAULT_API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    PROVIDER_NAME = "DeepSeek"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize DeepSeek provider with DeepSeek configuration."""
        super().__init__(config)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Create a chat completion using DeepSeek with Pydantic model support.

        DeepSeek doesn't support Pydantic parsing directly, so we convert
        Pydantic models to JSON format instructions.
        """
        # Handle Pydantic model conversion for DeepSeek
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to DeepSeek JSON format
                modified_messages = _convert_pydantic_to_deepseek_json(response_format, messages)
                kwargs["response_format"] = {"type": "json_object"}
                messages = modified_messages

        # Call the parent completion method
        return super().completion(model, messages, **kwargs)
