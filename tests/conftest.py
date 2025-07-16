import pytest

from any_llm.provider import ProviderName


@pytest.fixture(params=list(ProviderName), ids=lambda x: x.value)
def provider(request: pytest.FixtureRequest) -> ProviderName:
    return request.param  # type: ignore[no-any-return]
