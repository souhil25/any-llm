import json
from typing import Any


from pydantic import BaseModel


def _convert_pydantic_to_huggingface_json(
    pydantic_model: type[BaseModel], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert Pydantic model to HuggingFace-compatible JSON instructions.

    Following a similar pattern to the DeepSeek provider but adapted for HuggingFace.

    Returns:
        modified_messages
    """
    # Get the JSON schema from the Pydantic model
    schema = pydantic_model.model_json_schema()

    # Add JSON instructions to the last user message
    modified_messages = messages.copy()
    if modified_messages and modified_messages[-1]["role"] == "user":
        original_content = modified_messages[-1]["content"]
        json_instruction = f"""Answer the following question and format your response as a JSON object matching this schema:

Schema: {json.dumps(schema, indent=2)}

DO NOT return the schema itself. Instead, answer the question and put your answer in the correct JSON format.

For example, if the question asks for a name and you want to answer "Paris", return: {{"name": "Paris"}}

Question: {original_content}

Answer (as JSON):"""
        modified_messages[-1]["content"] = json_instruction
    else:
        msg = "Last message is not a user message"
        raise ValueError(msg)

    return modified_messages
