import os
from typing import Any, Optional

try:
    import boto3
except ImportError:
    msg = "boto3 is not installed. Please install it with `pip install any-llm-sdk[aws]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from any_llm.providers.aws.utils import _convert_response, _convert_kwargs, _convert_messages


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

            bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

            if credentials is None and bedrock_api_key is None:
                raise MissingApiKeyError(
                    provider_name="AWS", env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                )

        except Exception as e:
            if isinstance(e, MissingApiKeyError):
                raise
            # If any other error o  ccurs while checking credentials, treat as missing
            raise MissingApiKeyError(
                provider_name="AWS", env_var_name="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            ) from e

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using AWS Bedrock."""
        # Check credentials before creating client
        self._check_aws_credentials()

        # Create client if not already created
        if self.client is None:
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)  # type: ignore[no-untyped-call]

        system_message, formatted_messages = _convert_messages(messages)
        request_config = _convert_kwargs(kwargs)

        assert self.client is not None  # For mypy
        response = self.client.converse(
            modelId=model,
            messages=formatted_messages,
            system=system_message,
            **request_config,
        )

        # Convert to OpenAI format
        return _convert_response(response)
