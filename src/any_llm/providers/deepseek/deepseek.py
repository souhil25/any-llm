from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider


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