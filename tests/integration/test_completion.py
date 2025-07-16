import httpx
import pytest
from any_llm import completion, ProviderName
from any_llm.exceptions import MissingApiKeyError


def test_providers(provider: ProviderName, provider_model_map: dict[ProviderName, str]) -> None:
    """Test that all supported providers can be loaded successfully."""
    model_id = provider_model_map[provider]
    try:
        result = completion(f"{provider.value}/{model_id}", messages=[{"role": "user", "content": "Hello"}])
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        if provider == ProviderName.OLLAMA:
            pytest.skip("Ollama is not set up, skipping")
        raise
    assert result.choices[0].message.content is not None
