from any_llm.providers.openai.base import BaseOpenAIProvider


class FireworksProvider(BaseOpenAIProvider):
    """
    Fireworks AI Provider using BaseOpenAIProvider.

    Fireworks AI has an OpenAI-compatible API, so we can use the base OpenAI provider
    with just different configuration values.
    """

    # Fireworks-specific configuration
    DEFAULT_API_BASE = "https://api.fireworks.ai/inference/v1"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_NAME = "Fireworks"
