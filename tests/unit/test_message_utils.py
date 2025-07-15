import pytest

from any_llm.utils.message import convert_response_to_openai, _convert_usage_to_openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage


def test_basic_response_conversion() -> None:
    """Test basic response conversion with minimal required fields."""
    response_data = {
        "id": "chat-123",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [{"message": {"content": "Hello, world!", "role": "assistant"}, "finish_reason": "stop"}],
    }

    result = convert_response_to_openai(response_data)

    assert isinstance(result, ChatCompletion)
    assert result.id == "chat-123"
    assert result.model == "gpt-4"
    assert result.object == "chat.completion"
    assert result.created == 1234567890
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello, world!"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].index == 0


def test_response_with_usage_data() -> None:
    """Test response conversion with usage data included."""
    response_data = {
        "id": "chat-456",
        "model": "gpt-3.5-turbo",
        "created": 1234567890,
        "choices": [{"message": {"content": "Test response", "role": "assistant"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15},
    }

    result = convert_response_to_openai(response_data)

    assert result.usage is not None
    assert result.usage.completion_tokens == 10
    assert result.usage.prompt_tokens == 5
    assert result.usage.total_tokens == 15


def test_response_with_tool_calls() -> None:
    """Test response conversion with tool calls in the message."""
    response_data = {
        "id": "chat-789",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [
            {
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "get_weather", "arguments": '{"location": "San Francisco"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    result = convert_response_to_openai(response_data)

    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 1
    tool_call = result.choices[0].message.tool_calls[0]
    assert tool_call.id == "call_123"
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location": "San Francisco"}'


def test_response_with_multiple_choices() -> None:
    """Test response conversion with multiple choices."""
    response_data = {
        "id": "chat-multi",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [
            {"message": {"content": "Choice 1", "role": "assistant"}, "finish_reason": "stop"},
            {"message": {"content": "Choice 2", "role": "assistant"}, "finish_reason": "length"},
        ],
    }

    result = convert_response_to_openai(response_data)

    assert len(result.choices) == 2
    assert result.choices[0].index == 0
    assert result.choices[0].message.content == "Choice 1"
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[1].index == 1
    assert result.choices[1].message.content == "Choice 2"
    assert result.choices[1].finish_reason == "length"


def test_response_missing_optional_fields() -> None:
    """Test response conversion when optional fields are missing."""
    response_data = {
        "id": "chat-minimal",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [
            {
                "message": {
                    "content": "Minimal response"
                    # Missing 'role' - should default to 'assistant'
                }
                # Missing 'finish_reason' - should default to 'stop'
            }
        ],
        # Missing 'usage' - should be None
    }

    result = convert_response_to_openai(response_data)

    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage is None


def test_response_with_empty_tool_calls() -> None:
    """Test response conversion when tool_calls is empty or None."""
    response_data = {
        "id": "chat-empty-tools",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [
            {"message": {"content": "No tools used", "role": "assistant", "tool_calls": None}, "finish_reason": "stop"}
        ],
    }

    result = convert_response_to_openai(response_data)

    assert result.choices[0].message.tool_calls is None


def test_response_with_detailed_usage() -> None:
    """Test response conversion with detailed usage information."""
    response_data = {
        "id": "chat-detailed",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [{"message": {"content": "Detailed usage response", "role": "assistant"}, "finish_reason": "stop"}],
        "usage": {
            "completion_tokens": 20,
            "prompt_tokens": 10,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 15},
        },
    }

    result = convert_response_to_openai(response_data)

    assert result.usage is not None
    assert result.usage.completion_tokens == 20
    assert result.usage.prompt_tokens == 10
    assert result.usage.total_tokens == 30
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 5
    assert result.usage.completion_tokens_details is not None
    assert result.usage.completion_tokens_details.reasoning_tokens == 15


def test_response_with_malformed_tool_calls() -> None:
    """Test response conversion handles malformed tool calls gracefully."""
    response_data = {
        "id": "chat-malformed",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [
            {
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_valid", "function": {"name": "valid_function", "arguments": "{}"}},
                        {
                            # Missing 'id' field
                            "function": {"name": "missing_id_function", "arguments": "{}"}
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    result = convert_response_to_openai(response_data)

    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 2
    # First tool call should be valid
    assert result.choices[0].message.tool_calls[0].id == "call_valid"
    # Second tool call should handle missing id (defaults to "unknown")
    assert result.choices[0].message.tool_calls[1].id == "unknown"


def test_basic_usage_conversion() -> None:
    """Test basic usage conversion with required fields."""
    usage_data = {"completion_tokens": 100, "prompt_tokens": 50, "total_tokens": 150}

    result = _convert_usage_to_openai(usage_data)

    assert isinstance(result, CompletionUsage)
    assert result.completion_tokens == 100
    assert result.prompt_tokens == 50
    assert result.total_tokens == 150
    assert result.prompt_tokens_details is None
    assert result.completion_tokens_details is None


def test_usage_conversion_with_details() -> None:
    """Test usage conversion with optional detail fields."""
    usage_data = {
        "completion_tokens": 75,
        "prompt_tokens": 25,
        "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 10},
        "completion_tokens_details": {"reasoning_tokens": 20},
    }

    result = _convert_usage_to_openai(usage_data)

    assert result.completion_tokens == 75
    assert result.prompt_tokens == 25
    assert result.total_tokens == 100
    assert result.prompt_tokens_details is not None
    assert result.prompt_tokens_details.cached_tokens == 10
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.reasoning_tokens == 20


def test_usage_conversion_missing_optional_fields() -> None:
    """Test usage conversion when optional detail fields are missing."""
    usage_data = {
        "completion_tokens": 60,
        "prompt_tokens": 40,
        "total_tokens": 100,
        # Missing detail fields
    }

    result = _convert_usage_to_openai(usage_data)

    assert result.completion_tokens == 60
    assert result.prompt_tokens == 40
    assert result.total_tokens == 100
    assert result.prompt_tokens_details is None
    assert result.completion_tokens_details is None


def test_missing_required_fields_raises_error() -> None:
    """Test that missing required fields raise appropriate errors."""
    # Missing 'choices' field
    incomplete_data = {
        "id": "chat-incomplete",
        "model": "gpt-4",
        "created": 1234567890,
        # Missing 'choices'
    }

    with pytest.raises(KeyError):
        convert_response_to_openai(incomplete_data)


def test_empty_choices_list() -> None:
    """Test response conversion with empty choices list."""
    response_data = {"id": "chat-empty", "model": "gpt-4", "created": 1234567890, "choices": []}

    result = convert_response_to_openai(response_data)

    assert len(result.choices) == 0


def test_choice_missing_message() -> None:
    """Test that choices missing message field raise appropriate errors."""
    response_data = {
        "id": "chat-no-message",
        "model": "gpt-4",
        "created": 1234567890,
        "choices": [
            {
                "finish_reason": "stop"
                # Missing 'message'
            }
        ],
    }

    with pytest.raises(KeyError):
        convert_response_to_openai(response_data)
