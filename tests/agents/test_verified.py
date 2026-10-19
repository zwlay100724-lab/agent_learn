from pathlib import Path

import pytest

from minisweagent.agents import get_agent_class
from minisweagent.agents.verified import (
    VerifiedAgent,
    VerifiedAgentConfig,
    derive_task_state,
    parse_git_status,
)
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


def make_agent(tmp_path, outputs=None, env=None, **kwargs) -> VerifiedAgent:
    return VerifiedAgent(
        model=DeterministicModel(outputs=outputs or []),
        env=env or LocalEnvironment(cwd=str(tmp_path)),
        **{**BASE_CONFIG, "output_path": tmp_path / "run.traj.json", **kwargs},
    )


class CrashingVerifierEnv:
    """Delegates to LocalEnvironment, but raises for the verification command."""

    def __init__(self, cwd):
        self._env = LocalEnvironment(cwd=cwd)

    def execute(self, action, *args, **kwargs):
        if action.get("command") == "verify-crash":
            raise RuntimeError("verifier crashed")
        return self._env.execute(action, *args, **kwargs)

    def get_template_vars(self, **kwargs):
        return self._env.get_template_vars(**kwargs)

    def serialize(self):
        return self._env.serialize()


def log_path(tmp_path):
    return tmp_path / "run.traj.json.events.jsonl"


@pytest.mark.parametrize(
    ("exit_status", "expected"),
    [
        ("Submitted", "Unverified"),
        ("LimitsExceeded", "Incomplete"),
        ("TimeExceeded", "Incomplete"),
        ("RepeatedFormatError", "Incomplete"),
        ("RuntimeError", "Error"),
        (None, "Error"),
    ],
)
def test_derive_task_state(exit_status, expected):
    assert derive_task_state(exit_status) == expected


def test_add_messages_derives_task_state_for_exit_messages(tmp_path):
    message = make_agent(tmp_path).add_messages(
        {"role": "exit", "content": "", "extra": {"exit_status": "LimitsExceeded"}}
    )[0]
    assert message["extra"]["task_state"] == "Incomplete"


def test_pre_set_task_state_is_not_overwritten(tmp_path):
    message = make_agent(tmp_path).add_messages(
        {"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "task_state": "Succeeded"}}
    )[0]
    assert message["extra"]["task_state"] == "Succeeded"


def test_serialize_includes_state_and_empty_evidence(tmp_path):
    agent = make_agent(tmp_path)
    agent.add_messages({"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}})
    info = agent.serialize()["info"]
    assert (info["task_state"], info["verification"], info["delivery"]) == ("Unverified", None, None)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (" M a.py\n?? b.py\n", ["a.py", "b.py"]),
        ("R  old.py -> new.py\n", ["new.py"]),
        ("", []),
        ("ab\n", []),
    ],
)
def test_parse_git_status(output, expected):
    assert parse_git_status(output) == expected


def test_verified_agent_is_registered():
    assert get_agent_class("verified") is VerifiedAgent


def test_config_rejects_a_zero_attempt_budget():
    with pytest.raises(ValueError):
        VerifiedAgentConfig(system_template="s", instance_template="i", verify_max_attempts=0)


def test_successful_verification_marks_succeeded(tmp_path):
    agent = make_agent(tmp_path, [make_output("", [{"command": SUBMIT}])], verify_command="true")
    info = agent.run("task")

    assert (info["exit_status"], info["task_state"]) == ("Submitted", "Succeeded")
    verification = [e for e in read_events(log_path(tmp_path)) if e["event"] == "verification"]
    assert [(e["passed"], e["attempt"]) for e in verification] == [(True, 1)]


def test_failed_verification_feeds_back_and_retries(tmp_path):
    agent = make_agent(
        tmp_path,
        [
            make_output("", [{"command": SUBMIT}]),
            make_output("", [{"command": "touch marker"}]),
            make_output("", [{"command": SUBMIT}]),
        ],
        verify_command="test -f marker",
        verify_max_attempts=2,
    )
    info = agent.run("task")

    assert (info["exit_status"], info["task_state"]) == ("Submitted", "Succeeded")
    assert agent.n_calls == 3
    assert [e["attempt"] for e in read_events(log_path(tmp_path)) if e["event"] == "verification"] == [1, 2]
    assert "Verification failed" in agent.messages[3]["content"]


def test_no_verifier_accepts_as_unverified(tmp_path):
    agent = make_agent(tmp_path, [make_output("", [{"command": SUBMIT}])])
    info = agent.run("task")

    assert (info["exit_status"], info["task_state"]) == ("Submitted", "Unverified")
    assert not [e for e in read_events(log_path(tmp_path)) if e["event"] == "verification"]


def test_exhausted_budget_accepts_as_verification_failed(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}]), make_output("", [{"command": SUBMIT}])],
        verify_command="false",
        verify_max_attempts=2,
    )
    info = agent.run("task")

    assert (info["exit_status"], info["task_state"]) == ("Submitted", "VerificationFailed")
    verification = [e for e in read_events(log_path(tmp_path)) if e["event"] == "verification"]
    assert [(e["passed"], e["attempt"]) for e in verification] == [(False, 1), (False, 2)]


def test_feedback_disabled_accepts_immediately(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}])],
        verify_command="false",
        verify_feedback=False,
        verify_max_attempts=3,
    )
    info = agent.run("task")

    assert (info["task_state"], agent.n_calls) == ("VerificationFailed", 1)
    assert len([e for e in read_events(log_path(tmp_path)) if e["event"] == "verification"]) == 1


def test_verification_infrastructure_failure_accepts_as_unverified(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}])],
        env=CrashingVerifierEnv(str(tmp_path)),
        verify_command="verify-crash",
    )
    info = agent.run("task")

    assert (info["exit_status"], info["task_state"]) == ("Submitted", "Unverified")
    event = [e for e in read_events(log_path(tmp_path)) if e["event"] == "verification"][0]
    assert (event["passed"], event["exception_type"]) == (None, "RuntimeError")


def test_feedback_references_the_offloaded_output(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}]), make_output("", [{"command": SUBMIT}])],
        verify_command="python3 -c \"print('x' * 20000); raise SystemExit(1)\"",
        verify_max_attempts=2,
    )
    agent.run("task")

    event = [e for e in read_events(log_path(tmp_path)) if e["event"] == "verification"][0]
    assert Path(event["output_ref"]).exists()
    assert event["output_ref"] in agent.messages[3]["content"]


def test_run_resets_the_verification_budget(tmp_path):
    agent = make_agent(tmp_path, [make_output("", [{"command": SUBMIT}])], verify_command="true")
    agent._verification_attempts = 5
    agent.run("task")
    assert agent._verification_attempts == 1


def test_delivery_records_changed_files(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}])],
        verify_command="true",
        delivery_command="printf ' M a.py\\n?? b.py\\n'",
    )
    info = agent.run("task")

    assert info["delivery"]["changed_files"] == ["a.py", "b.py"]


def test_delivery_failure_records_none_without_failing(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}])],
        verify_command="true",
        delivery_command="false",
    )
    info = agent.run("task")

    assert (info["task_state"], info["delivery"]["changed_files"]) == ("Succeeded", None)


def test_delivery_is_skipped_when_disabled(tmp_path):
    agent = make_agent(
        tmp_path,
        [make_output("", [{"command": SUBMIT}])],
        verify_command="true",
        delivery_enabled=False,
    )
    info = agent.run("task")

    assert info["delivery"] is None
    assert not [e for e in read_events(log_path(tmp_path)) if e["event"] == "delivery"]


def test_delivery_runs_only_on_accepted_submission(tmp_path):
    agent = make_agent(
        tmp_path,
        [
            make_output("", [{"command": SUBMIT}]),
            make_output("", [{"command": "touch marker"}]),
            make_output("", [{"command": SUBMIT}]),
        ],
        verify_command="test -f marker",
        verify_max_attempts=2,
    )
    agent.run("task")

    assert len([e for e in read_events(log_path(tmp_path)) if e["event"] == "delivery"]) == 1