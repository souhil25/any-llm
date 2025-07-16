"""Custom exceptions for any-llm package."""


class MissingApiKeyError(Exception):
    """Exception raised when an API key is missing or not provided."""

    def __init__(self, provider_name: str, env_var_name: str) -> None:
        """Initialize the exception.

        Args:
            provider_name: Name of the provider (e.g., "OpenAI", "Google", "Mistral")
            env_var_name: Name of the environment variable that should contain the API key
            message: Optional custom message. If not provided, a default message will be used.
        """
        self.provider_name = provider_name
        self.env_var_name = env_var_name

        message = (
            f"No {provider_name} API key provided. "
            f"Please provide it in the config or set the {env_var_name} environment variable."
        )

        super().__init__(message)


class UnsupportedProviderError(Exception):
    """Exception raised when an unsupported provider is requested."""

    def __init__(self, provider_key: str, supported_providers: list[str]) -> None:
        """Initialize the exception.

        Args:
            provider_key: The provider key that was requested
            supported_providers: List of supported provider keys
        """
        self.provider_key = provider_key
        self.supported_providers = supported_providers

        message = f"'{provider_key}' is not a supported provider. Supported providers: {', '.join(supported_providers)}"

        super().__init__(message)
