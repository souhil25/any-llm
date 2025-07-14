from typing import Any

from llm_squid.provider import ProviderFactory
from llm_squid.tools import Tools


class Client:
    """Client for the providers."""

    def __init__(self, provider_configs: dict[str, Any] | None = None) -> None:
        """Initialize the Client."""
        if provider_configs is None:
            provider_configs = {}
        self.providers: dict[str, Any] = {}
        self.provider_configs = provider_configs
        self._chat: Chat | None = None
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize the providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key,
                config,
            )

    def _validate_provider_key(self, provider_key: str) -> str:
        """Validate the provider key."""
        supported_providers = ProviderFactory.get_supported_providers()

        if provider_key not in supported_providers:
            msg = f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. Make sure the model string is formatted correctly as 'provider/model'."
            raise ValueError(msg)

        return provider_key

    def configure(self, provider_configs: dict[str, Any] | None = None) -> None:
        """Configure the client."""
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        self._initialize_providers()  # NOTE: This will override existing provider instances.

    @property
    def chat(self) -> "Chat":
        """Get the chat instance."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat


class Chat:
    """Chat for the client."""

    def __init__(self, client: "Client") -> None:
        """Initialize the Chat."""
        self.client = client
        self._completions = Completions(self.client)

    @property
    def completions(self) -> "Completions":
        """Get the completions instance."""
        return self._completions


class Completions:
    """Completions for the client."""

    def __init__(self, client: "Client") -> None:
        """Initialize the Completions."""
        self.client = client

    def _extract_thinking_content(self, response: Any) -> Any:
        """Extract the thinking content from the response."""
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                content = message.content.strip()
                if content.startswith("<think>") and "</think>" in content:
                    # Extract content between think tags
                    start_idx = len("<think>")
                    end_idx = content.find("</think>")
                    thinking_content = content[start_idx:end_idx].strip()

                    # Store the thinking content
                    message.reasoning_content = thinking_content

                    # Remove the think tags from the original content
                    message.content = content[end_idx + len("</think>") :].strip()

        return response

    def _tool_runner(
        self,
        provider: Any,
        model_name: str,
        messages: list[Any],
        tools: Any,
        max_turns: int,
        **kwargs: Any,
    ) -> Any:
        """Run the tools."""
        # Handle tools validation and conversion
        if isinstance(tools, Tools):
            tools_instance = tools
            kwargs["tools"] = tools_instance.tools()
        else:
            # Check if passed tools are callable
            if not all(callable(tool) for tool in tools):
                msg = "One or more tools is not callable"
                raise ValueError(msg)
            tools_instance = Tools(tools)
            kwargs["tools"] = tools_instance.tools()

        turns = 0
        intermediate_responses = []  # Store intermediate responses
        intermediate_messages = []  # Store all messages including tool interactions

        while turns < max_turns:
            response = provider.chat_completions_create(model_name, messages, **kwargs)
            intermediate_responses.append(response)

            # Check if the response contains tool calls
            if (
                hasattr(response, "choices")
                and response.choices
                and hasattr(response.choices[0], "message")
                and hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                # Add the assistant's response to messages
                messages.append(response.choices[0].message)
                intermediate_messages.append(response.choices[0].message)

                # Execute the tools
                tool_results = tools_instance.execute_tool(
                    response.choices[0].message.tool_calls,
                )[1]
                messages.extend(tool_results)
                intermediate_messages.extend(tool_results)

                turns += 1
            else:
                # No tool calls, we're done
                break

        # Store intermediate messages in the final response
        response.choices[0].intermediate_messages = intermediate_messages[
            :-1
        ]  # Exclude final response
        response.choices[0].intermediate_messages = intermediate_messages
        return response

    def create(self, model: str, messages: list[Any], **kwargs: Any) -> Any:
        """Create a chat completion."""
        # Check that correct format is used
        if "/" not in model:
            msg = f"Invalid model format. Expected 'provider/model', got '{model}'"
            raise ValueError(msg)

        # Extract the provider key from the model identifier, e.g., "google/gemini-xx"
        provider_key, model_name = model.split("/", 1)

        # Validate if the provider is supported
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            msg = f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. Make sure the model string is formatted correctly as 'provider/model'."
            raise ValueError(msg)

        # Initialize provider if not already initialized
        if provider_key not in self.client.providers:
            config = self.client.provider_configs.get(provider_key, {})
            self.client.providers[provider_key] = ProviderFactory.create_provider(
                provider_key,
                config,
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            msg = f"Could not load provider for '{provider_key}'."
            raise ValueError(msg)

        # Extract tool-related parameters
        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.get("tools")

        # Check environment variable before allowing multi-turn tool execution
        if max_turns is not None and tools is not None:
            return self._tool_runner(
                provider,
                model_name,
                messages.copy(),
                tools,
                max_turns,
            )

        # Default behavior without tool execution
        # Delegate the chat completion to the correct provider's implementation
        response = provider.chat_completions_create(model_name, messages, **kwargs)
        return self._extract_thinking_content(response)
