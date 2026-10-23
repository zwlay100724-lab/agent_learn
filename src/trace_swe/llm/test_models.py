import logging
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from trace_swe.llm import GLOBAL_MODEL_STATS
from trace_swe.llm.utils.actions_toolcall import format_toolcall_observation_messages
from trace_swe.llm.utils.openai_multimodal import expand_multimodal_content


def _format_observation_messages(
    outputs: list[dict],
    *,
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Render execution outputs as user observation messages for the test model."""
    results = []
    for output in outputs:
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {
            "role": "user",
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results


def make_output(content: str, actions: list[dict], cost: float = 1.0) -> dict:
    """Helper to create an output dict for DeterministicModel.

    Args:
        content: The response content string
        actions: List of action dicts, e.g., [{"command": "echo hello"}]
        cost: Cost to report for this output (default 1.0)
    """
    return {
        "role": "assistant",
        "content": content,
        "extra": {"actions": actions, "cost": cost, "timestamp": time.time()},
    }


def make_toolcall_output(content: str | None, tool_calls: list[dict], actions: list[dict]) -> dict:
    """Helper to create a toolcall output dict for DeterministicToolcallModel.

    Args:
        content: Optional text content (can be None for tool-only responses)
        tool_calls: List of tool call dicts in OpenAI format
        actions: List of parsed action dicts, e.g., [{"command": "echo hello", "tool_call_id": "call_123"}]
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {"actions": actions, "cost": 1.0, "timestamp": time.time()},
    }


def _process_test_actions(actions: list[dict]) -> bool:
    """Process special test actions. Returns True if the query should be retried."""
    for action in actions:
        if "raise" in action:
            raise action["raise"]
        cmd = action.get("command", "")
        if cmd.startswith("/sleep "):
            time.sleep(float(cmd.split("/sleep ")[1]))
            return True
        if cmd.startswith("/warning"):
            logging.warning(cmd.split("/warning")[1])
            return True
    return False


class DeterministicModelConfig(BaseModel):
    outputs: list[dict]
    """List of exact output messages to return in sequence. Each dict should have 'role', 'content', and 'extra' (with 'actions')."""
    model_name: str = "deterministic"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class DeterministicModel:
    def __init__(self, **kwargs):
        """Initialize with a list of output messages to return in sequence."""
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        return _format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class DeterministicToolcallModelConfig(BaseModel):
    outputs: list[dict]
    """List of exact output messages with tool_calls to return in sequence."""
    model_name: str = "deterministic_toolcall"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class DeterministicToolcallModel:
    def __init__(self, **kwargs):
        """Initialize with a list of toolcall output messages to return in sequence."""
        self.config = DeterministicToolcallModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
