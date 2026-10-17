import json

from typer.testing import CliRunner

from minisweagent.models.test_models import make_output
from minisweagent.run.utilities.mini_extra import subcommands
from minisweagent.run.utilities.resume import app

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_trajectory(tmp_path):
    trajectory = tmp_path / "run.traj.json"
    data = {
        "info": {
            "task": "task",
            "model_stats": {"instance_cost": 0.0, "api_calls": 0},
            "config": {
                "agent": {
                    "system_template": "system",
                    "instance_template": "{{task}}",
                    "mode": "yolo",
                    "confirm_exit": False,
                    "output_path": str(trajectory),
                },
                "model": {
                    "outputs": [make_output("done", [{"command": SUBMIT}])],
                    "model_name": "deterministic",
                    "cost_per_call": 1.0,
                },
                "model_type": "minisweagent.models.test_models.DeterministicModel",
                "environment": {"cwd": str(tmp_path), "env": {}, "timeout": 30},
                "environment_type": "minisweagent.environments.local.LocalEnvironment",
            },
            "events": None,
        },
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}],
        "trajectory_format": "mini-swe-agent-1.1",
    }
    trajectory.write_text(json.dumps(data))
    return trajectory


def test_resume_cli_rebuilds_agent_from_trajectory(tmp_path):
    result = CliRunner().invoke(app, [str(make_trajectory(tmp_path))])
    assert result.exit_code == 0
    assert "Submitted" in result.output


def test_resume_cli_rejects_missing_file(tmp_path):
    result = CliRunner().invoke(app, [str(tmp_path / "nope.traj.json")])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_resume_is_registered_in_mini_extra():
    assert any("resume" in aliases for _, aliases, _ in subcommands)
