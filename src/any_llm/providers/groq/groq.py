import os
import json
from typing import Any

try:
    import groq
except ImportError:
    msg = "groq is not installed. Please install it with `pip install any-llm-sdk[groq]`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.base_framework import create_completion_from_response


class GroqProvider(Provider):
    """Groq Provider using the new response conversion utilities."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Groq provider."""
        if not config.api_key:
            config.api_key = os.getenv("GROQ_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Groq", "GROQ_API_KEY")

        self.client = groq.Groq(api_key=config.api_key)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Groq."""
        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to JSON schema format for Groq
                schema = response_format.model_json_schema()
                kwargs["response_format"] = {"type": "json_object"}

                # Add JSON instruction to the last user message (required by Groq)
                if messages and messages[-1]["role"] == "user":
                    original_content = messages[-1]["content"]
                    json_instruction = f"""
Please respond with a JSON object that matches the following schema:

{json.dumps(schema, indent=2)}

Return the JSON object only, no other text.

{original_content}
"""
                    messages[-1]["content"] = json_instruction

        # Clean messages (remove refusal field as per original implementation)
        cleaned_messages = []
        for message in messages:
            cleaned_message = message.copy()
            cleaned_message.pop("refusal", None)
            cleaned_messages.append(cleaned_message)

        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # type: ignore[arg-type]
                **kwargs,
            )

            # Convert to OpenAI format using the new utility
            return create_completion_from_response(
                response_data=response.model_dump(),
                model=model,
                provider_name="groq",
            )

        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e
