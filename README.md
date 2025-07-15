<p align="center">
  <picture>
    <img src="docs/images/any-llm-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-llm

[![Docs](https://github.com/mozilla-ai/any-llm/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-llm/actions/workflows/docs.yaml/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A single interface to use and evaluate different llm providers.

</div>

## [Documentation](https://mozilla-ai.github.io/any-llm/)

## [Supported Frameworks](https://mozilla-ai.github.io/any-llm/)

## Requirements

- Python 3.11 or newer

## Quickstart

Refer to [pyproject.toml](./pyproject.toml) for a list of the options available.
Update your pip install command to include the frameworks that you plan on using:

```bash
pip install 'any-llm'
```

Make sure you have the appropriate API key environment variable set for your provider

```bash
export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
```

```python
from any_llm import completion

# Simple completion without provider configuration
response = completion(
    model="mistral/mistral-small-latest",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```
