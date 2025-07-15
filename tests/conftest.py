import pytest

from any_llm.utils.provider import ProviderFactory


@pytest.fixture(params=list(ProviderFactory.get_supported_providers()), ids=lambda x: x)
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)
