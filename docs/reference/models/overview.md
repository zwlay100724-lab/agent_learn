# Models Overview

This project keeps one real model path: `LitellmModel` with native tool calling.
`DeterministicModel` is a test double.

| Class | Shortcut | Endpoint | Toolcalls | Description |
|-------|----------|----------|-----------|-------------|
| [`LitellmModel`](litellm.md) | `litellm` | `/completion` | ✅ | Uses [LiteLLM](https://docs.litellm.ai/docs/providers) to reach OpenAI, Anthropic, DeepSeek, and other providers |
| [`DeterministicModel`](test_models.md) | `deterministic` | N/A | ❌ | Returns predefined outputs for tests |

{% include-markdown "../../_footer.md" %}
