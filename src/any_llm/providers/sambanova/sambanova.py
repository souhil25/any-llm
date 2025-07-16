import json
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider


class SambanovaProvider(BaseOpenAIProvider):
    """
    SambaNova Provider implementation.

    This provider connects to SambaNova's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use SambaNova's configuration.

    Configuration:
    - api_key: SambaNova API key (can be set via SAMBANOVA_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to SambaNova's API)

    Example usage:
        config = ApiConfig(api_key="your-sambanova-api-key")
        provider = SambanovaProvider(config)
        response = provider.completion("your-model", messages=[...])
    """

    # SambaNova-specific configuration
    DEFAULT_API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    PROVIDER_NAME = "SambaNova"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize SambaNova provider with SambaNova configuration."""
        super().__init__(config)
        # Initialize instructor client for structured output
        self._initialize_instructor_client(config)

    def _initialize_instructor_client(self, config: ApiConfig) -> None:
        """Initialize instructor client for structured output."""
        try:
            import instructor
        except ImportError:
            raise ImportError(
                "The 'instructor' library is required for SambaNova structured output. "
                "Install it with: pip install instructor"
            )

        # Create OpenAI client with SambaNova configuration
        openai_client = OpenAI(
            base_url=config.api_base or self.DEFAULT_API_BASE,
            api_key=config.api_key,
        )

        # Wrap with instructor
        self.instructor_client = instructor.from_openai(openai_client)

    def _make_api_call(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
        """Make the API call to SambaNova service with instructor for structured output."""
        if "response_format" in kwargs:
            # Use instructor for structured output
            response_format = kwargs.pop("response_format")
            response = self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )
            # Convert instructor response to ChatCompletion format
            return self._convert_instructor_response(response, model)
        else:
            # Use standard OpenAI client for regular completions
            return self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

    def _convert_instructor_response(self, instructor_response: Any, model: str) -> ChatCompletion:
        """Convert instructor response to ChatCompletion format."""
        # For structured output, we need to create a mock ChatCompletion
        # that contains the structured response in the content
        # Convert the structured response to JSON string
        if hasattr(instructor_response, "model_dump"):
            content = json.dumps(instructor_response.model_dump())
        else:
            content = json.dumps(instructor_response)

        # Create a mock ChatCompletion response
        message = ChatCompletionMessage(
            role="assistant",
            content=content,
        )

        choice = Choice(
            finish_reason="stop",
            index=0,
            message=message,
        )

        return ChatCompletion(
            id="sambanova-instructor-response",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
        )
