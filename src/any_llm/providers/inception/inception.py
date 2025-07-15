from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider


class InceptionProvider(BaseOpenAIProvider):
    """
    Inception Labs Provider implementation.

    This provider connects to Inception Labs' API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use Inception's configuration.

    Configuration:
    - api_key: Inception API key (can be set via INCEPTION_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to Inception's API)

    Example usage:
        config = ApiConfig(api_key="your-inception-api-key")
        provider = InceptionProvider(config)
        response = provider.completion("your-model", messages=[...])
    """

    # Inception-specific configuration
    DEFAULT_API_BASE = "https://api.inceptionlabs.ai/v1"
    ENV_API_KEY_NAME = "INCEPTION_API_KEY"
    PROVIDER_NAME = "Inception"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Inception provider with Inception Labs configuration."""
        super().__init__(config) 