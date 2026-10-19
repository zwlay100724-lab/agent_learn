import pytest

from minisweagent.agents import get_agent_class
from minisweagent.agents.verified import (
    VerifiedAgent,
    VerifiedAgentConfig,
    derive_task_state,
    parse_git_status,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel

BASE_CONFIG = {
    "system_template": "system",
    "instance_template": "{{task}}",
    "mode": "yolo",
    "confirm_exit": False,
}


def make_agent(tmp_path, **kwargs) -> VerifiedAgent:
    return VerifiedAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(cwd=str(tmp_path)),
        **{**BASE_CONFIG, "output_path": tmp_path / "run.traj.json", **kwargs},
    )


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