from any_llm.provider import ApiConfig
from ..openai.base import BaseOpenAIProvider


class NebiusProvider(BaseOpenAIProvider):
    """
    Nebius AI Studio Provider implementation.

    This provider connects to Nebius AI Studio's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use Nebius's configuration.

    Configuration:
    - api_key: Nebius API key (can be set via NEBIUS_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to Nebius's API)

    Example usage:
        config = ApiConfig(api_key="your-nebius-api-key")
        provider = NebiusProvider(config)
        response = provider.completion("your-model", messages=[...])
    """

    # Nebius-specific configuration
    DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"
    ENV_API_KEY_NAME = "NEBIUS_API_KEY"
    PROVIDER_NAME = "Nebius"