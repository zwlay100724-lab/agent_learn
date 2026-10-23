import json
from pathlib import Path

from typer.testing import CliRunner

from trace_swe.cli.commands.resume import app
from trace_swe.cli.commands.trace_extra import subcommands
from trace_swe.llm.test_models import make_output

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_trajectory(tmp_path, agent_type="", agent_config=None):
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
                    **(agent_config or {}),
                },
                "agent_type": agent_type,
                "model": {
                    "outputs": [make_output("done", [{"command": SUBMIT}])],
                    "model_name": "deterministic",
                    "cost_per_call": 1.0,
                },
                "model_type": "trace_swe.llm.test_models.DeterministicModel",
                "environment": {"cwd": str(tmp_path), "env": {}, "timeout": 30},
                "environment_type": "trace_swe.execution.local.LocalEnvironment",
            },
            "events": None,
        },
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}],
        "trajectory_format": "trace-swe-1.1",
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


def test_resume_is_registered_in_trace_extra():
    assert any("resume" in aliases for _, aliases, _ in subcommands)


def test_resume_cli_restores_the_saved_agent_class(tmp_path):
    trajectory = make_trajectory(
        tmp_path,
        agent_type="trace_swe.agent.verified.VerifiedAgent",
        agent_config={"verify_command": "true"},
    )

    result = CliRunner().invoke(app, [str(trajectory)])

    assert result.exit_code == 0
    assert "Succeeded" in result.output
    events = [json.loads(line) for line in Path(f"{trajectory}.events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "verification" for e in events)
