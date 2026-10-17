from pathlib import Path

import pytest

from minisweagent.agents import get_agent_class
from minisweagent.agents.event_log import EventLogAgent
from minisweagent.agents.resume import ResumableAgent, ResumeBlocked, ResumeConfig
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.utils.event_log import read_events

BASE_CONFIG = {
    "system_template": "system",
    "instance_template": "{{task}}",
    "mode": "yolo",
    "confirm_exit": False,
}

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_agent(model, tmp_path, **kwargs) -> ResumableAgent:
    return ResumableAgent(
        model=model,
        env=LocalEnvironment(cwd=str(tmp_path)),
        **{**BASE_CONFIG, "output_path": tmp_path / "run.traj.json", **kwargs},
    )


def log_path(tmp_path):
    return tmp_path / "run.traj.json.events.jsonl"


def start_interrupted_run(tmp_path, outputs, steps, save_after=None):
    """Simulate a process that took `steps` steps and died without run()/run_end."""
    agent = make_agent(DeterministicModel(outputs=outputs), tmp_path)
    agent.extra_template_vars |= {"task": "task"}
    agent.messages = []
    agent.add_messages(
        agent.model.format_message(role="system", content=agent._render_template(agent.config.system_template)),
        agent.model.format_message(role="user", content=agent._render_template(agent.config.instance_template)),
    )
    for i in range(1, steps + 1):
        agent.step()
        if save_after is None or i <= save_after:
            agent.save(agent.config.output_path)
    agent.events.close()
    return agent


def test_resume_continues_from_a_clean_snapshot(tmp_path):
    start_interrupted_run(tmp_path, [make_output("one", [{"command": "echo one"}])], steps=1)

    agent = make_agent(DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]), tmp_path)
    info = agent.resume(tmp_path / "run.traj.json")

    assert (info["exit_status"], info["submission"]) == ("Submitted", "done\n")
    assert (agent.n_calls, agent.cost) == (2, 2.0)
    assert len(agent.messages) == 6
    assert agent.messages[2]["content"] == "one"


def test_resume_refuses_a_dirty_boundary(tmp_path):
    start_interrupted_run(
        tmp_path,
        [make_output("one", [{"command": "echo one"}]), make_output("two", [{"command": "echo two"}])],
        steps=2,
        save_after=1,
    )

    with pytest.raises(ResumeBlocked) as exc:
        make_agent(DeterministicModel(outputs=[]), tmp_path).resume(tmp_path / "run.traj.json")
    assert "echo two" in str(exc.value)


def test_resume_as_is_proceeds_past_a_dirty_boundary(tmp_path):
    start_interrupted_run(
        tmp_path,
        [make_output("one", [{"command": "echo one"}]), make_output("two", [{"command": "echo two"}])],
        steps=2,
        save_after=1,
    )

    agent = make_agent(
        DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]), tmp_path, resume_policy="as_is"
    )
    assert agent.resume(tmp_path / "run.traj.json")["exit_status"] == "Submitted"


def test_resume_refuses_a_finished_run(tmp_path):
    make_agent(DeterministicModel(outputs=[make_output("", [{"command": SUBMIT}])]), tmp_path).run("task")

    with pytest.raises(ResumeBlocked):
        make_agent(DeterministicModel(outputs=[]), tmp_path).resume(tmp_path / "run.traj.json")


def test_resume_records_a_run_resumed_event(tmp_path):
    start_interrupted_run(tmp_path, [make_output("one", [{"command": "echo one"}])], steps=1)
    make_agent(DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]), tmp_path).resume(
        tmp_path / "run.traj.json"
    )

    resumed = [e for e in read_events(log_path(tmp_path)) if e["event"] == "run_resumed"]
    assert [(e["clean"], e["last_event"]) for e in resumed] == [(True, "snapshot_written")]


def test_resume_continues_the_event_sequence(tmp_path):
    start_interrupted_run(tmp_path, [make_output("one", [{"command": "echo one"}])], steps=1)
    make_agent(DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]), tmp_path).resume(
        tmp_path / "run.traj.json"
    )

    events = read_events(log_path(tmp_path))
    keys = [(e["run_id"], e["seq"]) for e in events]
    assert len(keys) == len(set(keys))
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)


def test_resume_records_a_workspace_probe(tmp_path):
    start_interrupted_run(tmp_path, [make_output("one", [{"command": "echo one"}])], steps=1)
    make_agent(
        DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]),
        tmp_path,
        resume_check_command="echo probe",
    ).resume(tmp_path / "run.traj.json")

    probe = [e for e in read_events(log_path(tmp_path)) if e["event"] == "workspace_probe"][0]
    assert "probe" in probe["output_preview"]


def test_resume_config_keeps_event_log_fields():
    config = ResumeConfig(
        system_template="s", instance_template="i", event_log_output_preview_chars=10, resume_policy="as_is"
    )
    assert (config.event_log_output_preview_chars, config.resume_policy) == (10, "as_is")


def test_resumable_extends_event_log_agent():
    assert issubclass(ResumableAgent, EventLogAgent)
    assert get_agent_class("resumable") is ResumableAgent


def test_resume_offload_files_do_not_collide(tmp_path):
    large = "python3 -c \"print('x' * 20000)\""
    start_interrupted_run(tmp_path, [make_output("", [{"command": large}])], steps=1)

    agent = make_agent(
        DeterministicModel(outputs=[make_output("", [{"command": large}]), make_output("", [{"command": SUBMIT}])]),
        tmp_path,
    )
    agent.resume(tmp_path / "run.traj.json")

    refs = [e["output_ref"] for e in read_events(log_path(tmp_path)) if e["event"] == "action" and e["output_ref"]]
    assert len(refs) == 2 and len(set(refs)) == 2

    assert all(Path(ref).exists() for ref in refs)
