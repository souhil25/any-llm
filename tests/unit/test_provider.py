from pathlib import Path
import pytest

from any_llm.provider import ProviderFactory, ProviderName
from any_llm.exceptions import UnsupportedProviderError


def test_all_providers_in_enum() -> None:
    """Test that all provider directories are accounted for in the ProviderName enum."""
    # Get the path to the providers directory
    providers_dir = Path(__file__).parent.parent.parent / "src" / "any_llm" / "providers"

    # Get all provider directories (excluding __pycache__ and files)
    provider_dirs = []
    for item in providers_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            provider_dirs.append(item.name)

    # Get all enum values
    enum_values = [provider.value for provider in ProviderName]

    # Sort both lists for easier comparison
    provider_dirs.sort()
    enum_values.sort()

    # Check that all directories have corresponding enum values
    missing_from_enum = set(provider_dirs) - set(enum_values)
    missing_from_dirs = set(enum_values) - set(provider_dirs)

    assert not missing_from_enum, f"Provider directories missing from ProviderName enum: {missing_from_enum}"
    assert not missing_from_dirs, f"ProviderName enum values missing provider directories: {missing_from_dirs}"

    # Ensure they match exactly
    assert provider_dirs == enum_values, f"Provider directories {provider_dirs} don't match enum values {enum_values}"


def test_provider_enum_values_match_directory_names() -> None:
    """Test that enum values exactly match the provider directory names."""
    providers_dir = Path(__file__).parent.parent.parent / "src" / "any_llm" / "providers"

    # Get all provider directories
    actual_providers = set()
    for item in providers_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            actual_providers.add(item.name)

    # Get enum values
    enum_providers = {provider.value for provider in ProviderName}

    assert actual_providers == enum_providers, (
        f"Provider directories and enum values don't match!\n"
        f"In directories but not enum: {actual_providers - enum_providers}\n"
        f"In enum but not directories: {enum_providers - actual_providers}"
    )


def test_get_provider_enum_valid_provider() -> None:
    """Test get_provider_enum returns correct enum for valid provider."""
    provider_enum = ProviderFactory.get_provider_enum("openai")
    assert provider_enum == ProviderName.OPENAI


def test_get_provider_enum_invalid_provider() -> None:
    """Test get_provider_enum raises UnsupportedProviderError for invalid provider."""
    with pytest.raises(UnsupportedProviderError) as exc_info:
        ProviderFactory.get_provider_enum("invalid_provider")

    exception = exc_info.value
    assert exception.provider_key == "invalid_provider"
    assert isinstance(exception.supported_providers, list)
    assert len(exception.supported_providers) > 0
    assert "openai" in exception.supported_providers


def test_unsupported_provider_error_message() -> None:
    """Test UnsupportedProviderError has correct message format."""
    with pytest.raises(UnsupportedProviderError, match="'invalid_provider' is not a supported provider"):
        ProviderFactory.get_provider_enum("invalid_provider")


def test_unsupported_provider_error_attributes() -> None:
    """Test UnsupportedProviderError has correct attributes."""
    try:
        ProviderFactory.get_provider_enum("nonexistent")
    except UnsupportedProviderError as e:
        assert e.provider_key == "nonexistent"
        assert e.supported_providers == ProviderFactory.get_supported_providers()
        assert "Supported providers:" in str(e)
    else:
        pytest.fail("Expected UnsupportedProviderError to be raised")
