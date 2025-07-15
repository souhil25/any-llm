#!/usr/bin/env python3
"""Simple test to verify the custom exception works correctly."""

from any_llm.exceptions import MissingApiKeyError

# Test the custom exception
try:
    raise MissingApiKeyError("Test Provider", "TEST_API_KEY")
except MissingApiKeyError as e:
    print(f"✅ Custom exception works: {e}")
    print(f"   Provider: {e.provider_name}")
    print(f"   Env var: {e.env_var_name}")

# Test with custom message
try:
    raise MissingApiKeyError("Test Provider", "TEST_API_KEY", "Custom error message")
except MissingApiKeyError as e:
    print(f"✅ Custom message works: {e}")

print("All tests passed! 🎉")
