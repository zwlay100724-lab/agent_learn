import os
import subprocess
import sys
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


def make_agent(model, tmp_path, *, env=None, output_path=None, **kwargs) -> ResumableAgent:
    return ResumableAgent(
        model=model,
        env=env or LocalEnvironment(cwd=str(tmp_path)),
        **{**BASE_CONFIG, "output_path": output_path or tmp_path / "run.traj.json", **kwargs},
    )


def init_repo(path: Path) -> None:
    (path / "a.py").write_text("one\n")
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=eval", "-c", "user.email=eval@example.com", "commit", "-qm", "initial"],
    ):
        subprocess.run(args, cwd=path, check=True, capture_output=True)


def log_path(tmp_path):
    return tmp_path / "run.traj.json.events.jsonl"


def start_interrupted_run(tmp_path, outputs, steps, save_after=None, **kwargs):
    """Simulate a process that took `steps` steps and died without run()/run_end."""
    agent = make_agent(DeterministicModel(outputs=outputs), tmp_path, **kwargs)
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


def test_resume_preserves_token_totals_from_loaded_messages(tmp_path):
    first = make_output("one", [{"command": "echo one"}])
    first["extra"]["response"] = {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    start_interrupted_run(tmp_path, [first], steps=1)

    agent = make_agent(DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]), tmp_path)
    agent.resume(tmp_path / "run.traj.json")

    tokens = agent.serialize()["info"]["tokens"]
    assert (tokens["prompt_tokens"], tokens["n_calls_with_usage"]) == (10, 1)


def test_snapshot_records_a_workspace_fingerprint(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    init_repo(repo)
    start_interrupted_run(
        tmp_path,
        [make_output("one", [{"command": "echo two >> a.py"}])],
        steps=1,
        env=LocalEnvironment(cwd=str(repo)),
        output_path=out / "run.traj.json",
    )

    snapshot = [e for e in read_events(out / "run.traj.json.events.jsonl") if e["event"] == "snapshot_written"][0]
    assert (snapshot["workspace"]["kind"], bool(snapshot["workspace"]["tree"])) == ("git", True)


def test_resume_refuses_when_the_workspace_changed_since_the_snapshot(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    init_repo(repo)
    start_interrupted_run(
        tmp_path,
        [make_output("one", [{"command": "echo two >> a.py"}])],
        steps=1,
        env=LocalEnvironment(cwd=str(repo)),
        output_path=out / "run.traj.json",
    )
    (repo / "a.py").write_text("external change\n")

    agent = make_agent(
        DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]),
        tmp_path,
        env=LocalEnvironment(cwd=str(repo)),
        output_path=out / "run.traj.json",
    )
    with pytest.raises(ResumeBlocked, match="workspace changed"):
        agent.resume(out / "run.traj.json")


def test_resume_as_is_records_the_workspace_change(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir()
    out.mkdir()
    init_repo(repo)
    start_interrupted_run(
        tmp_path,
        [make_output("one", [{"command": "echo two >> a.py"}])],
        steps=1,
        env=LocalEnvironment(cwd=str(repo)),
        output_path=out / "run.traj.json",
    )
    (repo / "a.py").write_text("external change\n")

    agent = make_agent(
        DeterministicModel(outputs=[make_output("done", [{"command": SUBMIT}])]),
        tmp_path,
        env=LocalEnvironment(cwd=str(repo)),
        output_path=out / "run.traj.json",
        resume_policy="as_is",
    )
    info = agent.resume(out / "run.traj.json")

    assert info["exit_status"] == "Submitted"
    event = [e for e in read_events(out / "run.traj.json.events.jsonl") if e["event"] == "run_resumed"][0]
    assert (event["workspace_changed"], event["workspace"]["kind"]) == (True, "git")


@pytest.mark.parametrize(
    ("point", "expected_actions"),
    [("before_action", 0), ("after_action", 1), ("before_snapshot", 1)],
)
def test_resume_refuses_each_interruption_window_without_replaying(tmp_path, point, expected_actions):
    out = tmp_path / "run.traj.json"
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_data.fault_injection"],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ
        | {"FAULT_POINT": point, "FAULT_CWD": str(tmp_path), "FAULT_OUT": str(out), "MSWEA_SILENT_STARTUP": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17

    events = read_events(Path(f"{out}.events.jsonl"))
    last_snapshot = max(i for i, event in enumerate(events) if event["event"] == "snapshot_written")
    inflight = events[last_snapshot + 1 :]
    assert sum(1 for event in inflight if event["event"] == "action") == expected_actions
    assert inflight[-1]["event"] in ("llm_call", "action", "step_end")

    log_before = (tmp_path / "log.txt").read_text()
    agent = make_agent(DeterministicModel(outputs=[]), tmp_path, output_path=out)
    with pytest.raises(ResumeBlocked, match="not a snapshot"):
        agent.resume(out)
    assert (tmp_path / "log.txt").read_text() == log_before


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
