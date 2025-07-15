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
