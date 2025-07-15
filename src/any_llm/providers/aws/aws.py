import os
import json
from typing import Any, Optional, Literal

try:
    import boto3
    import botocore
except ImportError:
    msg = "boto3 is not installed. Please install it with `pip install any-llm-sdk[aws]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError


INFERENCE_PARAMETERS = ["maxTokens", "temperature", "topP", "stopSequences"]


def _convert_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Format the kwargs for AWS Bedrock."""
    kwargs = kwargs.copy()

    # Convert tools and remove from kwargs
    tool_config = _convert_tool_spec(kwargs)
    kwargs.pop("tools", None)  # Remove tools from kwargs if present

    # Prepare inference config
    inference_config = {key: kwargs[key] for key in INFERENCE_PARAMETERS if key in kwargs}

    additional_fields = {key: value for key, value in kwargs.items() if key not in INFERENCE_PARAMETERS}

    request_config = {
        "inferenceConfig": inference_config,
        "additionalModelRequestFields": additional_fields,
    }

    if tool_config is not None:
        request_config["toolConfig"] = tool_config

    return request_config


def _convert_tool_spec(kwargs: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert tool specifications to Bedrock format."""
    if "tools" not in kwargs:
        return None

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", " "),
                    "inputSchema": {"json": tool["function"]["parameters"]},
                }
            }
            for tool in kwargs["tools"]
        ]
    }
    return tool_config


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert messages to AWS Bedrock format."""
    # Handle system message
    system_message = []
    if messages and messages[0]["role"] == "system":
        system_message = [{"text": messages[0]["content"]}]
        messages = messages[1:]

    formatted_messages = []
    for message in messages:
        # Skip any additional system messages
        if message["role"] == "system":
            continue

        if message["role"] == "tool":
            bedrock_message = _convert_tool_result(message)
            if bedrock_message:
                formatted_messages.append(bedrock_message)
        elif message["role"] == "assistant":
            bedrock_message = _convert_assistant(message)
            if bedrock_message:
                formatted_messages.append(bedrock_message)
        else:  # user messages
            formatted_messages.append(
                {
                    "role": message["role"],
                    "content": [{"text": message["content"]}],
                }
            )

    return system_message, formatted_messages


def _convert_tool_result(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert OpenAI tool result format to AWS Bedrock format."""
    if message["role"] != "tool" or "content" not in message:
        return None

    tool_call_id = message.get("tool_call_id")
    if not tool_call_id:
        raise RuntimeError("Tool result message must include tool_call_id")

    try:
        content_json = json.loads(message["content"])
        content = [{"json": content_json}]
    except json.JSONDecodeError:
        content = [{"text": message["content"]}]

    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_call_id, "content": content}}],
    }


def _convert_assistant(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert OpenAI assistant format to AWS Bedrock format."""
    if message["role"] != "assistant":
        return None

    content = []

    if message.get("content"):
        content.append({"text": message["content"]})

    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            if tool_call["type"] == "function":
                try:
                    input_json = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    input_json = tool_call["function"]["arguments"]

                content.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": input_json,
                        }
                    }
                )

    return {"role": "assistant", "content": content} if content else None


def _convert_response(response: dict[str, Any]) -> ChatCompletion:
    """Convert AWS Bedrock response to OpenAI ChatCompletion format."""
    # Check if the model is requesting tool use
    if response.get("stopReason") == "tool_use":
        tool_calls = []
        for content in response["output"]["message"]["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool["toolUseId"],
                        type="function",
                        function=Function(
                            name=tool["name"],
                            arguments=json.dumps(tool["input"]),
                        ),
                    )
                )

        if tool_calls:
            message = ChatCompletionMessage(
                content=None,
                role="assistant",
                tool_calls=tool_calls,
            )

            choice = Choice(
                finish_reason="tool_calls",
                index=0,
                message=message,
            )

            usage = None
            if "usage" in response:
                usage_data = response["usage"]
                usage = CompletionUsage(
                    completion_tokens=usage_data.get("outputTokens", 0),
                    prompt_tokens=usage_data.get("inputTokens", 0),
                    total_tokens=usage_data.get("totalTokens", 0),
                )

            return ChatCompletion(
                id=response.get("id", ""),
                model=response.get("model", ""),
                object="chat.completion",
                created=response.get("created", 0),
                choices=[choice],
                usage=usage,
            )

    # Handle regular text response
    content = response["output"]["message"]["content"][0]["text"]

    # Map Bedrock stopReason to OpenAI finish_reason
    stop_reason = response.get("stopReason")
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    if stop_reason == "complete":
        finish_reason = "stop"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message = ChatCompletionMessage(
        content=content,
        role="assistant",
        tool_calls=None,
    )

    choice = Choice(
        finish_reason=finish_reason,
        index=0,
        message=message,
    )

    usage = None
    if "usage" in response:
        usage_data = response["usage"]
        usage = CompletionUsage(
            completion_tokens=usage_data.get("outputTokens", 0),
            prompt_tokens=usage_data.get("inputTokens", 0),
            total_tokens=usage_data.get("totalTokens", 0),
        )

    return ChatCompletion(
        id=response.get("id", ""),
        model=response.get("model", ""),
        object="chat.completion",
        created=response.get("created", 0),
        choices=[choice],
        usage=usage,
    )


class AwsProvider(Provider):
    """AWS Bedrock Provider using boto3."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize AWS Bedrock provider."""
        # AWS uses region from environment variables or default
        self.region_name = os.getenv("AWS_REGION", "us-east-1")

        # Store config for later use
        self.config = config

        # Don't create client during init to avoid test failures
        self.client: Optional[Any] = None

    def _check_aws_credentials(self) -> None:
        """Check if AWS credentials are available."""
        try:
            # Create a session to check if credentials are available
            session = boto3.Session()  # type: ignore[no-untyped-call, attr-defined]
            credentials = session.get_credentials()  # type: ignore[no-untyped-call]

            if credentials is None:
                raise MissingApiKeyError(
                    provider_name="AWS", env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                )

            # Try to get the credentials to ensure they're not expired/invalid
            _ = credentials.access_key
            _ = credentials.secret_key

        except Exception as e:
            if isinstance(e, MissingApiKeyError):
                raise
            # If any other error occurs while checking credentials, treat as missing
            raise MissingApiKeyError(
                provider_name="AWS", env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            ) from e

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a chat completion using AWS Bedrock."""
        # Check credentials before creating client
        self._check_aws_credentials()

        # Create client if not already created
        if self.client is None:
            try:
                self.client = boto3.client("bedrock-runtime", region_name=self.region_name)  # type: ignore[no-untyped-call]
            except Exception as e:
                raise RuntimeError(f"Failed to create AWS Bedrock client: {e}") from e

        system_message, formatted_messages = _convert_messages(messages)
        request_config = _convert_kwargs(kwargs)

        try:
            assert self.client is not None  # For mypy
            response = self.client.converse(
                modelId=model,
                messages=formatted_messages,
                system=system_message,
                **request_config,
            )

            # Convert to OpenAI format
            return _convert_response(response)

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                error_message = e.response["Error"]["Message"]
                raise RuntimeError(f"AWS Bedrock validation error: {error_message}") from e
            raise RuntimeError(f"AWS Bedrock API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"AWS Bedrock API error: {e}") from e
