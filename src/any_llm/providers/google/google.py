import os
import json
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    msg = "google-genai is not installed. Please install it with `pip install any-llm-sdk[google]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError

DEFAULT_TEMPERATURE = 0.7


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for Google GenAI."""
    kwargs = kwargs.copy()

    # Convert tools if present
    if "tools" in kwargs:
        kwargs["tools"] = _convert_tool_spec(kwargs["tools"])

    # Handle unsupported parameters
    unsupported_params = ["response_format", "parallel_tool_calls"]
    for param in unsupported_params:
        if param in kwargs:
            kwargs.pop(param)

    return kwargs


def _convert_tool_spec(openai_tools: list[dict[str, Any]]) -> list[types.Tool]:
    """Convert OpenAI tool specification to Google GenAI format."""
    function_declarations = []

    for tool in openai_tools:
        if tool.get("type") != "function":
            continue

        function = tool["function"]
        parameters_dict = {
            "type": "object",
            "properties": {
                param_name: {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                    **({"enum": param_info["enum"]} if "enum" in param_info else {}),
                }
                for param_name, param_info in function["parameters"]["properties"].items()
            },
            "required": function["parameters"].get("required", []),
        }

        function_declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=types.Schema(**parameters_dict),
            )
        )

    return [types.Tool(function_declarations=function_declarations)]


def _convert_messages(messages: list[dict[str, Any]]) -> list[types.Content]:
    """Convert messages to Google GenAI format."""
    formatted_messages = []

    for message in messages:
        if message["role"] == "system":
            # System messages are treated as user messages in GenAI
            parts = [types.Part.from_text(text=message["content"])]
            formatted_messages.append(types.Content(role="user", parts=parts))
        elif message["role"] == "user":
            parts = [types.Part.from_text(text=message["content"])]
            formatted_messages.append(types.Content(role="user", parts=parts))
        elif message["role"] == "assistant":
            if "tool_calls" in message and message["tool_calls"]:
                # Handle function calls
                tool_call = message["tool_calls"][0]  # Assuming single function call for now
                function_call = tool_call["function"]

                parts = [
                    types.Part.from_function_call(
                        name=function_call["name"], args=json.loads(function_call["arguments"])
                    )
                ]
            else:
                # Handle regular text messages
                parts = [types.Part.from_text(text=message["content"])]

            formatted_messages.append(types.Content(role="model", parts=parts))
        elif message["role"] == "tool":
            # Convert tool result to function response
            try:
                content_json = json.loads(message["content"])
                part = types.Part.from_function_response(name=message.get("name", "unknown"), response=content_json)
                formatted_messages.append(types.Content(role="function", parts=[part]))
            except json.JSONDecodeError:
                # If not JSON, treat as text
                part = types.Part.from_function_response(
                    name=message.get("name", "unknown"), response={"result": message["content"]}
                )
                formatted_messages.append(types.Content(role="function", parts=[part]))

    return formatted_messages


def _convert_response(response: Any) -> ChatCompletion:
    """Convert Google GenAI response to OpenAI ChatCompletion format."""
    # Check if the response contains function calls
    if (
        hasattr(response.candidates[0].content.parts[0], "function_call")
        and response.candidates[0].content.parts[0].function_call
    ):
        function_call = response.candidates[0].content.parts[0].function_call

        # Convert the function call arguments to a dictionary
        args_dict = {}
        if hasattr(function_call, "args") and function_call.args:
            for key, value in function_call.args.items():
                args_dict[key] = value

        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{hash(function_call.name)}",
                type="function",
                function=Function(name=function_call.name, arguments=json.dumps(args_dict)),
            )
        ]

        message = ChatCompletionMessage(
            content=None,
            role="assistant",
            tool_calls=tool_calls,
        )

        finish_reason = "tool_calls"
    else:
        # Handle regular text response
        content = response.candidates[0].content.parts[0].text
        message = ChatCompletionMessage(
            content=content,
            role="assistant",
            tool_calls=None,
        )

        finish_reason = "stop"

    # Create the choice
    choice = Choice(
        finish_reason=finish_reason,  # type: ignore
        index=0,
        message=message,
    )

    # Create usage information (extract if available)
    usage = CompletionUsage(
        completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0)
        if hasattr(response, "usage_metadata")
        else 0,
        prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0)
        if hasattr(response, "usage_metadata")
        else 0,
        total_tokens=getattr(response.usage_metadata, "total_token_count", 0)
        if hasattr(response, "usage_metadata")
        else 0,
    )

    # Build the final ChatCompletion object
    return ChatCompletion(
        id="google_genai_response",
        model="google/genai",
        object="chat.completion",
        created=0,
        choices=[choice],
        usage=usage,
    )


class GoogleProvider(Provider):
    def __init__(self, config: ApiConfig) -> None:
        """Initialize Google GenAI provider."""
        # Check if we should use Vertex AI or Gemini Developer API
        self.use_vertex_ai = os.getenv("GOOGLE_USE_VERTEX_AI", "false").lower() == "true"

        if self.use_vertex_ai:
            # Vertex AI configuration
            self.project_id = os.getenv("GOOGLE_PROJECT_ID")
            self.location = os.getenv("GOOGLE_REGION", "us-central1")

            if not self.project_id:
                raise MissingApiKeyError(
                    "Google Vertex AI",
                    "GOOGLE_PROJECT_ID",
                )

            # Initialize client for Vertex AI
            self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        else:
            # Gemini Developer API configuration
            # Use api_key from config if provided, otherwise fall back to environment variables
            api_key = getattr(config, "api_key", None) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            if not api_key:
                raise MissingApiKeyError(
                    "Google Gemini Developer API",
                    "GEMINI_API_KEY/GOOGLE_API_KEY",
                )

            # Initialize client for Gemini Developer API
            self.client = genai.Client(api_key=api_key)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using Google GenAI."""
        kwargs = _convert_kwargs(kwargs)

        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)

        # Convert messages to GenAI format
        formatted_messages = _convert_messages(messages)

        # Handle tools if provided
        tools = kwargs.get("tools")

        # Create generation config
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            tools=tools,
        )

        # Generate content using the client
        # For now, let's use a simple string-based approach
        content_text = ""

        if len(formatted_messages) == 1 and formatted_messages[0].role == "user":
            # Single user message
            parts = formatted_messages[0].parts
            if parts and hasattr(parts[0], "text"):
                content_text = parts[0].text or ""
            else:
                content_text = "Hello"  # fallback
        else:
            # Multiple messages - concatenate user messages for simplicity
            content_parts = []
            for msg in formatted_messages:
                if msg.role == "user" and msg.parts:
                    if hasattr(msg.parts[0], "text") and msg.parts[0].text:
                        content_parts.append(msg.parts[0].text)

            content_text = "\n".join(content_parts)
            if not content_text:
                content_text = "Hello"  # fallback

        response = self.client.models.generate_content(model=model, contents=content_text, config=generation_config)

        # Convert and return the response
        return _convert_response(response)
