from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider


class XaiProvider(BaseOpenAIProvider):
    """
    xAI Provider implementation.

    This provider connects to xAI's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use xAI's configuration.

    Configuration:
    - api_key: xAI API key (can be set via XAI_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to xAI's API)

    Example usage:
        config = ApiConfig(api_key="your-xai-api-key")
        provider = XaiProvider(config)
        response = provider.completion("your-model", messages=[...])
    """

    # xAI-specific configuration
    DEFAULT_API_BASE = "https://api.x.ai/v1"
    ENV_API_KEY_NAME = "XAI_API_KEY"
    PROVIDER_NAME = "xAI"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize xAI provider with xAI configuration."""
        super().__init__(config)
