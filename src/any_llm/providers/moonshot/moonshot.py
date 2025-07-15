from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider


class MoonshotProvider(BaseOpenAIProvider):
    """
    Moonshot AI Provider implementation.

    This provider connects to Moonshot AI's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use Moonshot AI's configuration.

    Configuration:
    - api_key: Moonshot AI API key (can be set via MOONSHOT_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to Moonshot AI's API)

    Example usage:
        config = ApiConfig(api_key="your-moonshot-api-key")
        provider = MoonshotProvider(config)
        response = provider.completion("moonshot-v1-8k", messages=[...])
    """

    # Moonshot AI-specific configuration
    DEFAULT_API_BASE = "https://api.moonshot.cn/v1"
    ENV_API_KEY_NAME = "MOONSHOT_API_KEY"
    PROVIDER_NAME = "Moonshot AI"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Moonshot AI provider with Moonshot AI configuration."""
        super().__init__(config)
