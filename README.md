# LLM Squid

A simple, function-based API for interacting with multiple LLM providers.

## Usage

### Basic Usage

```python
from llm_squid import completion

# Simple completion without provider configuration
response = completion(
    model="mistral/mistral-small-latest",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```
