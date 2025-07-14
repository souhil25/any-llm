import inspect
import json
from collections.abc import Callable
from typing import Any

from docstring_parser import parse
from pydantic import BaseModel, Field, ValidationError, create_model


class Tools:
    """Tools for the LLM."""

    def __init__(self, tools: list[Callable[..., Any]] | None = None) -> None:
        """Initialize the Tools."""
        self._tools: dict[str, dict[str, Any]] = {}
        if tools:
            for tool in tools:
                self._add_tool(tool)

    def _add_tool(
        self,
        func: Callable[..., Any],
        param_model: type[BaseModel] | None = None,
    ) -> None:
        """Add a tool to the Tools."""
        if param_model:
            tool_spec = self._convert_to_tool_spec(func, param_model)
        else:
            tool_spec, param_model = self.__infer_from_signature(func)

        self._tools[func.__name__] = {
            "function": func,
            "param_model": param_model,
            "spec": tool_spec,
        }

    def tools(self, tool_format: str = "openai") -> list[dict[str, Any]]:
        """Return tools in the specified format (default OpenAI)."""
        if tool_format == "openai":
            return self.__convert_to_openai_format()
        msg = f"Unknown format: {tool_format}"
        raise ValueError(msg)

    def _convert_to_tool_spec(
        self,
        func: Callable[..., Any],
        param_model: type[BaseModel],
    ) -> dict[str, Any]:
        """Convert a function and its parameter model to a tool specification."""
        # Get the function's docstring
        func_doc = inspect.getdoc(func)
        description = func_doc if func_doc else f"Execute {func.__name__}"

        # Create the tool specification
        tool_spec: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": param_model.model_json_schema(),
            },
        }

        # Extract parameter descriptions from docstring
        param_descriptions = self.__extract_param_descriptions(func)
        tool_function = tool_spec.get("function", {})
        if not tool_function.get("parameters"):
            msg = "Tool function parameters are missing. Please ensure the tool function has parameters."
            raise ValueError(msg)
        parameters: dict[str, Any] = tool_function["parameters"]
        properties: dict[str, Any] = parameters.get("properties", {})

        # Handle case where param_model might be None or doesn't have proper iteration
        if param_model and hasattr(param_model, "__annotations__"):
            annotations = param_model.__annotations__
            for param_name in annotations.keys():
                if param_name in properties and param_name in param_descriptions:
                    properties[param_name]["description"] = param_descriptions[
                        param_name
                    ]

        return tool_spec

    def __extract_param_descriptions(self, func: Callable[..., Any]) -> dict[str, str]:
        """Extract parameter descriptions from function docstring."""
        func_doc = inspect.getdoc(func)
        if not func_doc:
            return {}

        parsed = parse(func_doc)
        param_descriptions = {}

        for param in parsed.params:
            if param.arg_name and param.description:
                param_descriptions[param.arg_name] = param.description

        return param_descriptions

    def __infer_from_signature(
        self,
        func: Callable[..., Any],
    ) -> tuple[dict[str, Any], type[BaseModel]]:
        """Infer tool specification from function signature."""
        sig = inspect.signature(func)
        field_definitions: dict[str, Any] = {}
        param_descriptions = self.__extract_param_descriptions(func)

        for param_name, param in sig.parameters.items():
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

            # Check if parameter has default value
            default_value = (
                param.default if param.default != inspect.Parameter.empty else ...
            )

            # Get description from docstring
            description = param_descriptions.get(param_name, f"Parameter {param_name}")

            # Create field definition using proper pydantic syntax
            if default_value is not ...:
                field_definitions[param_name] = (
                    param_type,
                    Field(default=default_value, description=description),
                )
            else:
                field_definitions[param_name] = (
                    param_type,
                    Field(description=description),
                )

        # Create a dynamic Pydantic model
        param_model = create_model(f"{func.__name__}Params", **field_definitions)

        tool_spec = self._convert_to_tool_spec(func, param_model)
        return tool_spec, param_model

    def __convert_to_openai_format(self) -> list[dict[str, Any]]:
        """Convert tools to OpenAI format."""
        return [tool_data["spec"] for tool_data in self._tools.values()]

    def results_to_messages(
        self,
        results: list[Any],
        message: Any,
    ) -> list[dict[str, Any]]:
        """Convert tool results to messages."""
        messages = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for i, tool_call in enumerate(message.tool_calls):
                result = results[i] if i < len(results) else "No result"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    },
                )
        return messages

    def execute(self, tool_calls: list[Any]) -> list[Any]:
        """Execute a list of tool calls."""
        results = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name in self._tools:
                func = self._tools[func_name]["function"]
                param_model = self._tools[func_name]["param_model"]

                try:
                    # Validate parameters using Pydantic model
                    validated_params = param_model(**args)
                    result = func(**validated_params.model_dump())
                    results.append(result)
                except ValidationError as e:
                    error_msg = f"Validation error for {func_name}: {e!s}"
                    results.append(error_msg)
                except Exception as e:
                    error_msg = f"Error executing {func_name}: {e!s}"
                    results.append(error_msg)
            else:
                results.append(f"Unknown tool: {func_name}")

        return results

    def execute_tool(
        self,
        tool_calls: list[Any],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Execute tool calls and return results and messages."""
        results = self.execute(tool_calls)

        # Convert results to messages format
        messages = []
        for i, tool_call in enumerate(tool_calls):
            result = results[i] if i < len(results) else "No result"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                },
            )

        return results, messages
