# Based on https://github.com/andrewyng/aisuite/blob/main/aisuite/utils/tools.py
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import ProviderFactory, ApiConfig


def completion(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
    """Create a chat completion.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'mistral/mistral-small')
        messages: List of messages for the conversation
        **kwargs: Additional parameters including:
            - tools: List of tools or Tools instance
            - max_turns: Maximum number of tool execution turns
            - api_key, api_base, etc.
            - Other provider-specific parameters

    Returns:
        The completion response from the provider

    """
    # Check that correct format is used
    if "/" not in model:
        msg = f"Invalid model format. Expected 'provider/model', got '{model}'"
        raise ValueError(msg)

    # Extract the provider key from the model identifier, e.g., "mistral/mistral-small"
    provider_key_str, model_name = model.split("/", 1)

    # Validate that neither provider nor model name is empty
    if not provider_key_str or not model_name:
        msg = f"Invalid model format. Expected 'provider/model', got '{model}'"
        raise ValueError(msg)

    # Convert string to ProviderName enum and validate
    provider_key = ProviderFactory.get_provider_enum(provider_key_str)

    # Create provider instance
    config: dict[str, str] = {}
    if "api_key" in kwargs:
        config["api_key"] = str(kwargs.pop("api_key"))
    if "api_base" in kwargs:
        config["api_base"] = str(kwargs.pop("api_base"))
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    return provider.completion(model_name, messages, **kwargs)
