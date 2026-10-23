# Python bindings

!!! abstract "Overview"

    This page shows the most basic example of how to use mini-SWE-agent as a Python library.
    For more advanced usage, subclassing, and mix & match of components, see [subclassing and more](../advanced/cookbook.md).

## Hello world

```python
import logging

from trace_swe.agent.default import DefaultAgent
from trace_swe.llm import get_model
from trace_swe.execution.local import LocalEnvironment

logging.basicConfig(level=logging.DEBUG)
task = "Write a hello world program"
model_name = "anthropic/claude-sonnet-4-5-20250929"

agent = DefaultAgent(
    get_model(input_model_name=model_name),
    LocalEnvironment(),
)

# Run the agent
agent.run(task)
```

{% include-markdown "../_footer.md" %}
