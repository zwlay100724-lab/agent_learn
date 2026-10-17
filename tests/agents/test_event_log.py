import time
from pathlib import Path

from minisweagent.agents import get_agent_class
from minisweagent.agents.event_log import EventLogAgent, EventLogAgentConfig
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.utils.event_log import read_events

BASE_CONFIG = {
    "system_template": "system",
    "instance_template": "{{task}}",
    "mode": "yolo",
    "confirm_exit": False,
}

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_agent(model, tmp_path, **kwargs) -> EventLogAgent:
    return EventLogAgent(
        model=model,
        env=LocalEnvironment(cwd=str(tmp_path)),
        **{**BASE_CONFIG, "output_path": tmp_path / "run.traj.json", **kwargs},
    )


def log_path(tmp_path):
    return tmp_path / "run.traj.json.events.jsonl"


def test_writes_ordered_events(tmp_path):
    agent = make_agent(
        DeterministicModel(
            outputs=[make_output("look", [{"command": "echo hi"}]), make_output("done", [{"command": SUBMIT}])]
        ),
        tmp_path,
    )
    assert agent.run("task")["exit_status"] == "Submitted"

    events = read_events(log_path(tmp_path))
    assert [e["event"] for e in events] == [
        "run_start",
        "step_start",
        "llm_call",
        "action",
        "step_end",
        "snapshot_written",
        "step_start",
        "llm_call",
        "action",
        "step_end",
        "snapshot_written",
        "run_end",
    ]
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert {e["run_id"] for e in events} == {events[0]["run_id"]}


class _SlowModel(DeterministicModel):
    def query(self, messages, **kwargs):
        time.sleep(0.02)
        return super().query(messages, **kwargs)


def test_records_per_step_timing(tmp_path):
    agent = make_agent(
        _SlowModel(outputs=[make_output("", [{"command": "sleep 0.02"}]), make_output("", [{"command": SUBMIT}])]),
        tmp_path,
    )
    agent.run("task")

    events = read_events(log_path(tmp_path))
    llm_calls = [e for e in events if e["event"] == "llm_call"]
    actions = [e for e in events if e["event"] == "action"]
    first_step = [e for e in events if e["event"] == "step_end"][0]
    assert llm_calls[0]["duration_s"] >= 0.015
    assert actions[0]["duration_s"] >= 0.015
    assert first_step["duration_s"] >= llm_calls[0]["duration_s"] + actions[0]["duration_s"]


class _BilledFormatErrorModel(DeterministicModel):
    def query(self, messages, **kwargs):
        raise FormatError({"role": "user", "content": "bad", "extra": {"cost": 0.25}})


def test_format_error_records_cost_and_outcome(tmp_path):
    agent = make_agent(_BilledFormatErrorModel(outputs=[], cost_per_call=0.25), tmp_path)
    assert agent.run("task")["exit_status"] == "RepeatedFormatError"

    events = read_events(log_path(tmp_path))
    calls = [e for e in events if e["event"] == "llm_call"]
    assert all(e["success"] is False and e["cost"] == 0.25 for e in calls)
    assert [e["outcome"] for e in events if e["event"] == "step_end"] == ["format_error"] * 3


def test_action_output_is_hashed_not_stored(tmp_path):
    agent = make_agent(
        DeterministicModel(
            outputs=[
                make_output("", [{"command": "python3 -c \"print('x' * 2000)\""}]),
                make_output("", [{"command": SUBMIT}]),
            ]
        ),
        tmp_path,
    )
    agent.run("task")

    action = [e for e in read_events(log_path(tmp_path)) if e["event"] == "action"][0]
    assert (action["output_chars"], action["output_truncated"], len(action["output_preview"])) == (2001, True, 500)
    assert action["output_ref"] is None
    assert "x" * 600 not in log_path(tmp_path).read_text()


def test_submission_event_chain(tmp_path):
    agent = make_agent(DeterministicModel(outputs=[make_output("", [{"command": SUBMIT}])]), tmp_path)
    agent.run("task")

    events = read_events(log_path(tmp_path))
    action, run_end = events[3], events[-1]
    assert (action["event"], action["submitted"], action["success"]) == ("action", True, True)
    assert (run_end["exit_status"], run_end["success"], run_end["submission_chars"]) == ("Submitted", True, 5)


def test_derives_log_path_and_records_it_in_trajectory(tmp_path):
    agent = make_agent(DeterministicModel(outputs=[make_output("", [{"command": SUBMIT}])]), tmp_path)
    agent.run("task")

    assert log_path(tmp_path).exists()
    assert (tmp_path / "run.traj.json").exists()
    assert agent.serialize()["info"]["events"]["path"] == str(log_path(tmp_path))


def test_disabled_without_paths(tmp_path):
    agent = EventLogAgent(
        model=DeterministicModel(outputs=[make_output("", [{"command": SUBMIT}])]),
        env=LocalEnvironment(cwd=str(tmp_path)),
        **{**BASE_CONFIG, "output_path": None},
    )
    assert agent.run("task")["exit_status"] == "Submitted"
    assert not list(tmp_path.iterdir())


def test_config_keeps_interactive_fields():
    config = EventLogAgentConfig(system_template="s", instance_template="i", mode="yolo", confirm_exit=False)
    assert (config.mode, config.confirm_exit) == ("yolo", False)


def test_no_behavior_change_vs_interactive_agent(tmp_path):
    def outputs():
        return [make_output("look", [{"command": "echo hi"}]), make_output("done", [{"command": SUBMIT}])]

    config = {**BASE_CONFIG, "output_path": tmp_path / "run.traj.json"}
    logged = EventLogAgent(
        model=DeterministicModel(outputs=outputs()), env=LocalEnvironment(cwd=str(tmp_path)), **config
    )
    plain = InteractiveAgent(
        model=DeterministicModel(outputs=outputs()), env=LocalEnvironment(cwd=str(tmp_path)), **config
    )

    logged_info, plain_info = logged.run("task"), plain.run("task")
    assert (logged_info["exit_status"], logged_info["submission"]) == (
        plain_info["exit_status"],
        plain_info["submission"],
    )
    assert (logged.n_calls, logged.cost, len(logged.messages)) == (plain.n_calls, plain.cost, len(plain.messages))


def test_agent_shorthand_is_registered():
    assert get_agent_class("event_log") is EventLogAgent


LARGE = "python3 -c \"print('x' * 20000)\""


def large_output_agent(tmp_path, **kwargs):
    template = (
        "{% if output.output_path is defined %}<path>{{ output.output_path }}</path>{% endif %}{{ output.output }}"
    )
    return make_agent(
        DeterministicModel(
            outputs=[make_output("", [{"command": LARGE}]), make_output("", [{"command": SUBMIT}])],
            observation_template=template,
        ),
        tmp_path,
        **kwargs,
    )


def test_large_output_is_offloaded_and_referenced(tmp_path):
    agent = large_output_agent(tmp_path)
    agent.run("task")

    action = [e for e in read_events(log_path(tmp_path)) if e["event"] == "action"][0]
    stored = Path(action["output_ref"])
    assert stored.exists() and len(stored.read_text()) == 20001
    assert str(stored) in agent.messages[3]["content"]


def test_small_output_is_not_offloaded(tmp_path):
    agent = make_agent(
        DeterministicModel(outputs=[make_output("", [{"command": "echo hi"}]), make_output("", [{"command": SUBMIT}])]),
        tmp_path,
    )
    agent.run("task")

    action = [e for e in read_events(log_path(tmp_path)) if e["event"] == "action"][0]
    assert action["output_ref"] is None
    assert not Path(f"{log_path(tmp_path)}.outputs").exists()


def test_offload_can_be_disabled(tmp_path):
    agent = large_output_agent(tmp_path, output_offload_enabled=False)
    agent.run("task")

    action = [e for e in read_events(log_path(tmp_path)) if e["event"] == "action"][0]
    assert action["output_ref"] is None
    assert not Path(f"{log_path(tmp_path)}.outputs").exists()


def test_offload_is_disabled_for_non_local_environments(tmp_path):
    class RemoteEnvironment:
        def __init__(self):
            self._env = LocalEnvironment(cwd=str(tmp_path))

        def execute(self, *args, **kwargs):
            return self._env.execute(*args, **kwargs)

        def get_template_vars(self, **kwargs):
            return self._env.get_template_vars(**kwargs)

        def serialize(self):
            return self._env.serialize()

    agent = EventLogAgent(
        model=DeterministicModel(
            outputs=[make_output("", [{"command": LARGE}]), make_output("", [{"command": SUBMIT}])]
        ),
        env=RemoteEnvironment(),
        **{**BASE_CONFIG, "output_path": tmp_path / "run.traj.json"},
    )
    agent.run("task")

    action = [e for e in read_events(log_path(tmp_path)) if e["event"] == "action"][0]
    assert action["output_ref"] is None


def test_offload_truncates_at_max_bytes(tmp_path):
    agent = large_output_agent(tmp_path, output_offload_max_bytes=100)
    agent.run("task")

    action = [e for e in read_events(log_path(tmp_path)) if e["event"] == "action"][0]
    content = Path(action["output_ref"]).read_text()
    assert content.startswith("x" * 100)
    assert "truncated at 100 bytes" in content
