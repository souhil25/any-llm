import httpx
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
    "together": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "xai": "xai-3-70b-instruct",
    "inception": "inception-3-70b-instruct",
    "nebius": "nebius-3-70b-instruct",
    "ollama": "llama3.1:8b",
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
