from any_llm.provider import ProviderFactory, ApiConfig, Provider


def test_all_providers_can_be_loaded(provider: str) -> None:
    """Test that all supported providers can be loaded successfully.

    This test uses the provider fixture which iterates over all providers
    returned by ProviderFactory.get_supported_providers(). It verifies that:
    1. Each provider can be imported and instantiated
    2. The created instance is actually a Provider
    3. No ImportError or other exceptions are raised during loading

    This test will automatically include new providers when they're added
    without requiring any code changes.
    """
    # Try to create the provider with empty config
    # This should not raise any ImportError or other loading exceptions
    provider_instance = ProviderFactory.create_provider(provider, ApiConfig())

    # Verify that the created instance is actually a Provider
    assert isinstance(provider_instance, Provider), f"Provider {provider} did not create a valid Provider instance"

    # Verify the provider has the required completion method
    assert hasattr(provider_instance, "completion"), f"Provider {provider} does not have a completion method"
    assert callable(provider_instance.completion), f"Provider {provider} completion is not callable"


def test_all_providers_can_be_loaded_with_config(provider: str) -> None:
    """Test that all supported providers can be loaded with sample config parameters.

    This test verifies that providers can handle common configuration parameters
    like api_key and api_base without throwing errors during instantiation.
    """
    # Sample config that might be passed to any provider
    sample_config = ApiConfig(api_key="test_key", api_base="https://test.example.com")

    # Try to create the provider with sample config
    # Providers should handle unknown config parameters gracefully
    provider_instance = ProviderFactory.create_provider(provider, sample_config)

    # Verify that the created instance is actually a Provider
    assert isinstance(provider_instance, Provider), (
        f"Provider {provider} did not create a valid Provider instance with config"
    )


def test_provider_factory_can_create_all_supported_providers() -> None:
    """Test that ProviderFactory can create instances of all providers it claims to support."""
    supported_providers = ProviderFactory.get_supported_providers()

    for provider_name in supported_providers:
        # Should be able to create each supported provider
        provider_instance = ProviderFactory.create_provider(provider_name, ApiConfig())

        # Each should be a valid Provider instance
        assert isinstance(provider_instance, Provider), f"Failed to create valid Provider instance for {provider_name}"
