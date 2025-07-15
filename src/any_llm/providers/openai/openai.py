from any_llm.provider import ApiConfig
from .base import BaseOpenAIProvider


class OpenaiProvider(BaseOpenAIProvider):
    """
    OpenAI Provider implementation.

    This provider connects to OpenAI's API using the official OpenAI Python client.
    It extends BaseOpenAIProvider to use OpenAI's default configuration.

    Configuration:
    - api_key: OpenAI API key (can be set via OPENAI_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to OpenAI's API)

    Example usage:
        config = ApiConfig(api_key="sk-...")
        provider = OpenaiProvider(config)
        response = provider.completion("gpt-4", messages=[...])
    """

    # OpenAI-specific configuration
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    ENV_API_KEY_NAME = "OPENAI_API_KEY"
    PROVIDER_NAME = "OpenAI"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize OpenAI provider with standard OpenAI configuration."""
        super().__init__(config)
