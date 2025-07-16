import asyncio
import httpx
import pytest
from any_llm import completion, acompletion, ProviderName
from any_llm.exceptions import MissingApiKeyError
from openai.types.chat.chat_completion import ChatCompletion


def test_streaming_completion(provider: ProviderName, provider_model_map: dict[ProviderName, str]) -> None:
    """Test that streaming completion works for supported providers."""
    pytest.skip("Skipping streaming completion test")
    model_id = provider_model_map[provider]
    try:
        result = completion(
            f"{provider.value}/{model_id}", 
            messages=[{"role": "user", "content": "Say 'Hello World' in 10 words or less"}],
            stream=True,
            temperature=0.1
        )
        
        # Verify the response is still a valid ChatCompletion object
        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content is not None
        assert len(result.choices) > 0
        assert hasattr(result, 'model')
        assert hasattr(result, 'id')
        assert hasattr(result, 'created')
        
        # Verify content is reasonable (should contain some form of greeting)
        content = result.choices[0].message.content.lower()
        assert any(word in content for word in ['hello', 'hi', 'greetings', 'world'])
        
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError):
        if provider == ProviderName.OLLAMA:
            pytest.skip("Ollama is not set up, skipping")
        raise