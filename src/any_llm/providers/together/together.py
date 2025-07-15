from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider


class TogetherProvider(BaseOpenAIProvider):
    """
    Together AI Provider implementation.

    This provider connects to Together AI's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use Together AI's configuration.

    Configuration:
    - api_key: Together AI API key (can be set via TOGETHER_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to Together AI's API)

    Example usage:
        config = ApiConfig(api_key="your-together-api-key")
        provider = TogetherProvider(config)
        response = provider.completion("meta-llama/Llama-2-7b-chat-hf", messages=[...])
    """

    # Together AI-specific configuration
    DEFAULT_API_BASE = "https://api.together.xyz/v1"
    ENV_API_KEY_NAME = "TOGETHER_API_KEY"
    PROVIDER_NAME = "Together AI"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Together AI provider with Together AI configuration."""
        super().__init__(config)
