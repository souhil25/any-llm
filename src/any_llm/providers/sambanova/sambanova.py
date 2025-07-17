from typing import Any

try:
    import instructor
except ImportError:
    msg = "instructor is not installed. Please install it with `pip install any-llm-sdk[sambanova]`"
    raise ImportError(msg)

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from any_llm.provider import ApiConfig, convert_instructor_response
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
        self._initialize_client(config)
        self._initialize_instructor_client(config)

    def _initialize_instructor_client(self, config: ApiConfig) -> None:
        """Initialize instructor client for structured output."""
        # Create OpenAI client with SambaNova configuration
        openai_client = OpenAI(
            base_url=config.api_base or self.DEFAULT_API_BASE,
            api_key=config.api_key,
        )

        # Wrap with instructor
        self.instructor_client = instructor.from_openai(openai_client)

    def completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
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
            return convert_instructor_response(response, model, "sambanova")
        else:
            # Use standard OpenAI client for regular completions
            return self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
