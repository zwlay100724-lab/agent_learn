import json

from typer.testing import CliRunner

from minisweagent.run.utilities.events import app, summarize
from minisweagent.run.utilities.mini_extra import subcommands


def _events():
    return [
        {"event": "run_start", "run_id": "r", "seq": 0, "step": 0, "mono": 100.0},
        {"event": "step_start", "run_id": "r", "seq": 1, "step": 1},
        {"event": "llm_retry", "run_id": "r", "seq": 2, "step": 1, "attempt": 1},
        {"event": "llm_call", "run_id": "r", "seq": 3, "step": 1, "duration_s": 2.0, "cost": 0.5, "attempts": 2},
        {"event": "action", "run_id": "r", "seq": 4, "step": 1, "duration_s": 0.5},
        {"event": "step_end", "run_id": "r", "seq": 5, "step": 1, "duration_s": 3.0, "outcome": "ok"},
        {"event": "step_start", "run_id": "r", "seq": 6, "step": 2},
        {"event": "llm_call", "run_id": "r", "seq": 7, "step": 2, "duration_s": 1.0, "cost": 0.25, "attempts": 1},
        {
            "event": "step_end",
            "run_id": "r",
            "seq": 8,
            "step": 2,
            "duration_s": 2.0,
            "outcome": "interrupted",
            "exit_status": "Submitted",
        },
        {"event": "run_end", "run_id": "r", "seq": 9, "step": 0, "duration_s": 6.0, "exit_status": "Submitted"},
    ]


def test_summarize_aggregates_timings_retries_and_failures():
    summary = summarize(_events())
    assert (summary["exit_status"], summary["incomplete"], summary["wall_s"]) == ("Submitted", False, 6.0)
    assert (summary["llm_s"], summary["env_s"], summary["retry_count"], summary["max_attempts"]) == (3.0, 0.5, 1, 2)
    assert summary["cost"] == 0.75
    assert [row["outcome"] for row in summary["steps"]] == ["ok", "interrupted"]
    assert summary["failures"] == []


def test_summarize_reports_failed_steps_but_not_submission():
    events = [
        {"event": "run_start", "run_id": "r", "seq": 0, "step": 0},
        {"event": "step_end", "run_id": "r", "seq": 1, "step": 1, "duration_s": 1.0, "outcome": "exception"},
        {"event": "step_end", "run_id": "r", "seq": 2, "step": 2, "duration_s": 1.0, "outcome": "format_error"},
        {
            "event": "step_end",
            "run_id": "r",
            "seq": 3,
            "step": 3,
            "duration_s": 1.0,
            "outcome": "interrupted",
            "exit_status": "Submitted",
        },
        {"event": "run_end", "run_id": "r", "seq": 4, "step": 0, "duration_s": 3.0, "exit_status": "Submitted"},
    ]
    assert [f["step"] for f in summarize(events)["failures"]] == [1, 2]


def test_summarize_marks_incomplete_run():
    summary = summarize(_events()[:-1])
    assert summary["incomplete"] is True
    assert (summary["last_event"], summary["exit_status"]) == ("step_end", "Submitted")


def test_summarize_reports_task_state():
    events = _events()
    events[-1]["task_state"] = "Succeeded"
    assert summarize(events)["task_state"] == "Succeeded"


def test_events_cli_prefers_task_state(tmp_path):
    events = _events()
    events[-1]["task_state"] = "VerificationFailed"
    path = tmp_path / "run.events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))

    result = CliRunner().invoke(app, [str(path)])
    assert result.exit_code == 0
    assert "VerificationFailed" in result.output


def test_events_cli_reads_jsonl(tmp_path):
    path = tmp_path / "run.events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in _events()))

    result = CliRunner().invoke(app, [str(path)])
    assert result.exit_code == 0
    assert "Submitted" in result.output
    assert "retries=1" in result.output


def test_events_cli_resolves_trajectory_path(tmp_path):
    log = tmp_path / "run.events.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in _events()))
    traj = tmp_path / "run.traj.json"
    traj.write_text(json.dumps({"info": {"events": {"path": str(log)}}, "messages": []}))

    result = CliRunner().invoke(app, [str(traj)])
    assert result.exit_code == 0
    assert "Submitted" in result.output


def test_events_cli_reports_missing_log(tmp_path):
    traj = tmp_path / "x.traj.json"
    traj.write_text(json.dumps({"info": {}}))

    result = CliRunner().invoke(app, [str(traj)])
    assert result.exit_code != 0
    assert "No event log found" in result.output


def test_events_is_registered_in_mini_extra():
    assert any("events" in aliases for _, aliases, _ in subcommands)
