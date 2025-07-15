import httpx
from pydantic import BaseModel
import pytest
from any_llm import completion
from any_llm.exceptions import MissingApiKeyError


# Use small models for testing to make sure they work
provider_model_map = {
    "mistral": "mistral-small-latest",
    "anthropic": "claude-3-5-sonnet-20240620",
    "deepseek": "deepseek-chat",
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.0-flash-001",
    "moonshot": "moonshot-v1-8k",
    "sambanova": "sambanova-7b-instruct",
    "together": "meta-llama/Meta-Llama-3-8B-Instruct",
    "xai": "xai-3-70b-instruct",
    "inception": "inception-3-70b-instruct",
    "nebius": "nebius-3-70b-instruct",
    "ollama": "llama3.2:3b",
    "azure": "gpt-4o",
    "cohere": "command-r-20240215",
    "cerebras": "llama3.1-8b",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",  # You must have novita enabled in your hf account to use this model
    "aws": "amazon.titan-text-001",
    "watsonx": "google/gemini-2.0-flash-001",
    "fireworks": "meta-llama/Meta-Llama-3-8B-Instruct",
    "groq": "llama-3.1-8b-instant",
}


def test_providers(provider: str) -> None:
    """Test that all supported providers can be loaded successfully."""
    model_id = provider_model_map[provider]
    try:
        result = completion(f"{provider}/{model_id}", messages=[{"role": "user", "content": "Hello"}])
    except MissingApiKeyError:
        pytest.skip(f"{provider} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        if provider == "ollama":
            pytest.skip("Ollama is not set up, skipping")
        raise
    assert result.choices[0].message.content is not None


def test_response_format(provider: str) -> None:
    """Test that all supported providers can be loaded successfully."""
    if provider == "anthropic":
        pytest.skip("Anthropic does not support response_format")
        return
    model_id = provider_model_map[provider]

    class ResponseFormat(BaseModel):
        name: str

    prompt = "What is the capital of France?"
    try:
        result = completion(
            f"{provider}/{model_id}", messages=[{"role": "user", "content": prompt}], response_format=ResponseFormat
        )
        assert result.choices[0].message.content is not None
        output = ResponseFormat.model_validate_json(result.choices[0].message.content)
        assert output.name == "Paris"
    except MissingApiKeyError:
        pytest.skip(f"{provider} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        if provider == "ollama":
            pytest.skip("Ollama is not set up, skipping")
        raise
