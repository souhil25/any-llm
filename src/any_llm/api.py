from typing import Any, Optional, List, Union, Callable

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from any_llm.provider import ProviderFactory, ApiConfig
from any_llm.tools import prepare_tools


def completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    # Common parameters with explicit types
    tools: Optional[List[Union[dict[str, Any], Callable[..., Any]]]] = None,
    tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    max_turns: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    # Generation control
    stream: Optional[bool] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    # Provider configuration
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    user: Optional[str] = None,
    # Additional provider-specific parameters
    **kwargs: Any,
) -> ChatCompletion:
    """Create a chat completion.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'mistral/mistral-small')
        messages: List of messages for the conversation
        tools: List of tools for tool calling. Can be Python callables or OpenAI tool format dicts
        tool_choice: Controls which tools the model can call
        max_turns: Maximum number of tool execution turns
        temperature: Controls randomness in the response (0.0 to 2.0)
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        response_format: Format specification for the response
        stream: Whether to stream the response
        n: Number of completions to generate
        stop: Stop sequences for generation
        presence_penalty: Penalize new tokens based on presence in text
        frequency_penalty: Penalize new tokens based on frequency in text
        seed: Random seed for reproducible results
        api_key: API key for the provider
        api_base: Base URL for the provider API
        timeout: Request timeout in seconds
        user: Unique identifier for the end user
        **kwargs: Additional provider-specific parameters

    Returns:
        The completion response from the provider

    """
    # Check that correct format is used
    if "/" not in model:
        msg = f"Invalid model format. Expected 'provider/model', got '{model}'"
        raise ValueError(msg)

    if stream:
        raise ValueError("Streaming is not yet supported")

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
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    # Build kwargs with explicit parameters
    completion_kwargs = kwargs.copy()
    if tools is not None:
        completion_kwargs["tools"] = prepare_tools(tools)
    if tool_choice is not None:
        completion_kwargs["tool_choice"] = tool_choice
    if max_turns is not None:
        completion_kwargs["max_turns"] = max_turns
    if temperature is not None:
        completion_kwargs["temperature"] = temperature
    if top_p is not None:
        completion_kwargs["top_p"] = top_p
    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        completion_kwargs["response_format"] = response_format
    if stream is not None:
        completion_kwargs["stream"] = stream
    if n is not None:
        completion_kwargs["n"] = n
    if stop is not None:
        completion_kwargs["stop"] = stop
    if presence_penalty is not None:
        completion_kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        completion_kwargs["frequency_penalty"] = frequency_penalty
    if seed is not None:
        completion_kwargs["seed"] = seed
    if timeout is not None:
        completion_kwargs["timeout"] = timeout
    if user is not None:
        completion_kwargs["user"] = user

    return provider.completion(model_name, messages, **completion_kwargs)
