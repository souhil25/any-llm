from contextlib import contextmanager
from unittest.mock import patch, Mock

from any_llm.provider import ApiConfig
from any_llm.providers.anthropic.anthropic import AnthropicProvider


@contextmanager
def mock_anthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.anthropic.anthropic.Anthropic") as mock_anthropic,
        patch("any_llm.providers.anthropic.anthropic._convert_kwargs", return_value={}),
        patch("any_llm.providers.anthropic.anthropic._convert_response"),
    ):
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = Mock()
        yield mock_anthropic


def test_anthropic_client_created_with_api_key_and_api_base() -> None:
    """Test that Anthropic client is created with api_key and api_base as base_url when provided."""
    api_key = "test-api-key"
    custom_endpoint = "https://custom-anthropic-endpoint"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key, api_base=custom_endpoint))
        provider._make_api_call("model-id", [{"role": "user", "content": "Hello"}])

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=custom_endpoint)


def test_anthropic_client_created_without_api_base() -> None:
    """Test that Anthropic client is created with None base_url when api_base is not provided."""
    api_key = "test-api-key"

    with mock_anthropic_provider() as mock_anthropic:
        provider = AnthropicProvider(ApiConfig(api_key=api_key))
        provider._make_api_call("model-id", [{"role": "user", "content": "Hello"}])

        mock_anthropic.assert_called_once_with(api_key=api_key, base_url=None)
