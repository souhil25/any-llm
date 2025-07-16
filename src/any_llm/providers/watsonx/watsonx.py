import os
from typing import Any

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore[attr-defined]
except ImportError:
    msg = "ibm-watsonx-ai is not installed. Please install it with `pip install any-llm-sdk[watsonx]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert Watsonx response to OpenAI ChatCompletion format."""
    choice_data = response["choices"][0]
    message_data = choice_data["message"]

    # Create the message
    message = ChatCompletionMessage(
        content=message_data.get("content"),
        role=message_data.get("role", "assistant"),
        tool_calls=None,  # Watsonx doesn't seem to support tool calls in the aisuite implementation
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


class WatsonxProvider(Provider):
    """IBM Watsonx Provider using the official IBM Watsonx AI SDK."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Watsonx provider."""
        # Get configuration from config or environment variables
        self.service_url = config.api_base or os.getenv("WATSONX_SERVICE_URL")
        self.api_key = config.api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")

        # Only validate API key during instantiation for consistency with other providers
        if not self.api_key:
            raise MissingApiKeyError("Watsonx", "WATSONX_API_KEY")

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Watsonx."""
        # Validate required configuration at runtime
        if not self.service_url:
            raise ValueError(
                "Missing WatsonX service URL. Please provide it in the config or set the WATSONX_SERVICE_URL environment variable."
            )
        if not self.project_id:
            raise ValueError(
                "Missing WatsonX project ID. Please provide it in the config or set the WATSONX_PROJECT_ID environment variable."
            )

        # Create ModelInference instance
        model_inference = ModelInference(
            model_id=model,
            credentials=Credentials(
                api_key=self.api_key,
                url=self.service_url,
            ),
            project_id=self.project_id,
        )

        # Make the API call
        response = model_inference.chat(
            messages=messages,
            params=kwargs,
        )

        # Convert to OpenAI format
        return _convert_response(response)
