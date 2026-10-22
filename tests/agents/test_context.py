import json
from pathlib import Path

from minisweagent.agents import get_agent_class
from minisweagent.agents.context import ContextManagedAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicToolcallModel,
    make_output,
    make_toolcall_output,
)
from minisweagent.utils.event_log import read_events

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_agent(tmp_path, outputs, model_class=DeterministicModel, **kwargs):
    return ContextManagedAgent(
        model=model_class(outputs=outputs),
        env=LocalEnvironment(cwd=str(tmp_path)),
        system_template="system",
        instance_template="{{task}}",
        mode="yolo",
        confirm_exit=False,
        cost_limit=0,
        step_limit=0,
        output_path=tmp_path / "run.traj.json",
        **kwargs,
    )


def scripted_agent(tmp_path, count, **kwargs):
    outputs = [make_output("", [{"command": f"echo obs-{i}"}]) for i in range(count)]
    outputs.append(make_output("", [{"command": SUBMIT}]))
    return make_agent(tmp_path, outputs, **kwargs)


def log_path(tmp_path):
    return tmp_path / "run.traj.json.events.jsonl"


def observations(agent):
    return [m for m in agent.messages if m.get("extra", {}).get("observation")]


def toolcall_output(index, command):
    tool_call = {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return make_toolcall_output("", [tool_call], [{"command": command, "tool_call_id": f"call_{index}"}])


def test_observation_messages_are_tagged(tmp_path):
    agent = scripted_agent(tmp_path, 2, context_max_observations=10)
    agent.run("task")
    assert len(observations(agent)) >= 2


def test_old_observations_are_elided(tmp_path):
    agent = scripted_agent(tmp_path, 5, context_max_observations=2, context_elide_min_chars=10)
    agent.run("task")
    assert [bool(message["extra"].get("elided")) for message in observations(agent)] == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_elided_message_points_at_the_full_output(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_max_observations=1, context_elide_min_chars=10)
    agent.run("task")

    stub = observations(agent)[0]["content"]
    path = stub.split("full output: ")[1].rstrip("]")
    assert "obs-0" in Path(path).read_text()


def test_small_observations_are_not_elided(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_max_observations=1, context_elide_min_chars=10_000)
    agent.run("task")
    assert not any(message["extra"].get("elided") for message in observations(agent))


def test_critical_messages_survive(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_max_observations=1, context_elide_min_chars=10)
    agent.run("task")

    assert agent.messages[0]["role"] == "system" and "elided" not in agent.messages[0].get("extra", {})
    assert agent.messages[1]["role"] == "user" and "elided" not in agent.messages[1].get("extra", {})
    assert all("elided" not in message.get("extra", {}) for message in agent.messages if message["role"] == "assistant")
    assert agent.messages[-1]["role"] == "exit" and "elided" not in agent.messages[-1].get("extra", {})


def test_elision_is_idempotent(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_max_observations=1, context_elide_min_chars=10)
    agent.run("task")
    before = [message.get("content") for message in observations(agent)]
    events_before = len([e for e in read_events(log_path(tmp_path)) if e["event"] == "context_elide"])

    agent._compact_context()

    assert [message.get("content") for message in observations(agent)] == before
    assert len([e for e in read_events(log_path(tmp_path)) if e["event"] == "context_elide"]) == events_before


def test_disabled_keeps_all_observations(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_elide_enabled=False)
    agent.run("task")
    assert not any(message["extra"].get("elided") for message in observations(agent))


def test_context_elide_event_is_recorded(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_max_observations=1, context_elide_min_chars=10)
    agent.run("task")
    elisions = [e for e in read_events(log_path(tmp_path)) if e["event"] == "context_elide"]
    assert sum(event["count"] for event in elisions) == 2


def test_serialize_reports_context_stats(tmp_path):
    agent = scripted_agent(tmp_path, 3, context_max_observations=1, context_elide_min_chars=10)
    agent.run("task")
    context = agent.serialize()["info"]["context"]
    assert context["elided_observations"] == 2
    assert context["elided_chars"] > 0


def test_toolcall_observations_are_elided(tmp_path):
    outputs = [toolcall_output(index, f"echo obs-{index}") for index in range(3)]
    outputs.append(toolcall_output(9, SUBMIT))
    agent = make_agent(
        tmp_path,
        outputs,
        model_class=DeterministicToolcallModel,
        context_max_observations=1,
        context_elide_min_chars=10,
    )
    agent.run("task")

    elided = [message for message in observations(agent) if message["extra"].get("elided")]
    assert elided and "full output:" in elided[0]["content"]


def test_context_agent_is_registered():
    assert get_agent_class("context") is ContextManagedAgent
