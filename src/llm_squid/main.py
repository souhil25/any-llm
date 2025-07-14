# Based on https://github.com/andrewyng/aisuite/blob/main/aisuite/utils/tools.py
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from llm_squid.utils import ProviderFactory


def completion(model: str, messages: list[ChatCompletionMessage], **kwargs: dict[str, Any]) -> ChatCompletion:
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
    provider_key, model_name = model.split("/", 1)

    # Validate if the provider is supported
    supported_providers = ProviderFactory.get_supported_providers()
    if provider_key not in supported_providers:
        msg = f"{provider_key} is not a supported provider. Supported providers: {supported_providers}. Make sure the model string is formatted correctly as 'provider/model'."
        raise ValueError(msg)

    # Create provider instance
    config = {}
    if "api_key" in kwargs:
        config["api_key"] = kwargs.pop("api_key")
    if "api_base" in kwargs:
        config["api_base"] = kwargs.pop("api_base")
    if "api_version" in kwargs:
        config["api_version"] = kwargs.pop("api_version")

    provider = ProviderFactory.create_provider(provider_key, config)

    return provider.completion(model_name, messages, **kwargs)
