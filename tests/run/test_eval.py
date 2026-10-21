import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minisweagent.run.utilities.eval import (
    OUTCOMES,
    _p95,
    _pct_change,
    _stats,
    aggregate,
    app,
    build_report,
    build_run,
    classify_outcome,
    compare,
    discover,
    group_runs,
)
from minisweagent.run.utilities.mini_extra import subcommands


def make_run(outcome="Succeeded", **kwargs):
    run = {
        "path": "",
        "group": "",
        "outcome": outcome,
        "task_state": None,
        "exit_status": None,
        "cost": None,
        "n_calls": None,
        "n_steps": None,
        "wall_s": None,
        "llm_s": None,
        "env_s": None,
        "retry_count": None,
        "has_event_log": False,
        "event_log_error": None,
        "verification_attempts": None,
        "delivery_changed_files": None,
        "error": None,
    }
    run.update(kwargs)
    return run


@pytest.mark.parametrize(
    ("task_state", "exit_status", "expected"),
    [
        ("Succeeded", "Submitted", "Succeeded"),
        ("VerificationFailed", "Submitted", "VerificationFailed"),
        ("Unverified", "Submitted", "Unverified"),
        ("Incomplete", "Submitted", "Incomplete"),
        ("Error", "Submitted", "Error"),
        ("Bogus", "Submitted", "Error"),
        (None, "Submitted", "Unverified"),
        (None, "LimitsExceeded", "Incomplete"),
        (None, "TimeExceeded", "Incomplete"),
        (None, "RepeatedFormatError", "Incomplete"),
        (None, None, "Incomplete"),
        (None, "RuntimeError", "Error"),
    ],
)
def test_classify_outcome(task_state, exit_status, expected):
    assert classify_outcome(task_state, exit_status) == expected


def test_stats_ignore_none_and_report_coverage():
    stats = _stats([1.0, None, 3.0])
    assert (stats["total"], stats["mean"], stats["median"], stats["p95"], stats["n"]) == (4.0, 2.0, 2.0, 3.0, 2)


def test_stats_all_missing_returns_none():
    stats = _stats([None, None])
    assert (stats["total"], stats["mean"], stats["median"], stats["p95"], stats["n"]) == (None, None, None, None, 0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0], 1.0),
        ([1.0, 10.0], 10.0),
        (list(map(float, range(1, 11))), 10.0),
        ([], None),
    ],
)
def test_p95_uses_nearest_rank(values, expected):
    assert _p95(values) == expected


def test_aggregate_uses_strict_success():
    runs = [make_run(outcome) for outcome in OUTCOMES]
    summary = aggregate(runs)

    assert (summary["n_runs"], summary["successes"], summary["success_rate"]) == (5, 1, 0.2)
    assert summary["buckets"] == dict.fromkeys(OUTCOMES, 1)
    assert sum(summary["buckets"].values()) == summary["n_runs"]


def test_aggregate_retry_rate_uses_event_coverage():
    runs = [make_run(retry_count=1, has_event_log=True), make_run(retry_count=3, has_event_log=True), make_run()]
    assert aggregate(runs)["retries"] == {"total": 4, "rate": 2.0, "n": 2}


def write_trajectory(root, name="run.traj.json", info=None, messages=None, **extra):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"info": info or {}, "messages": messages or [], **extra}))
    return path


def write_events(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events))


def sample_events(run_id="r"):
    return [
        {"event": "run_start", "run_id": run_id, "seq": 0, "step": 0, "mono": 100.0},
        {"event": "step_start", "run_id": run_id, "seq": 1, "step": 1},
        {"event": "llm_call", "run_id": run_id, "seq": 2, "step": 1, "duration_s": 2.0, "cost": 0.5, "attempts": 1},
        {"event": "action", "run_id": run_id, "seq": 3, "step": 1, "duration_s": 0.5},
        {"event": "step_end", "run_id": run_id, "seq": 4, "step": 1, "duration_s": 3.0, "outcome": "ok"},
        {"event": "run_end", "run_id": run_id, "seq": 5, "step": 0, "duration_s": 4.0, "exit_status": "Submitted"},
    ]


def test_discover_recurses_and_sorts(tmp_path):
    for name in ("b.traj.json", "a/x.traj.json", "a.traj.json"):
        write_trajectory(tmp_path, name)
    paths = discover(tmp_path)
    assert sorted(p.name for p in paths) == ["a.traj.json", "b.traj.json", "x.traj.json"]
    assert paths == sorted(paths)


def test_discover_ignores_preds_json(tmp_path):
    (tmp_path / "preds.json").write_text("{}")
    write_trajectory(tmp_path, "a.traj.json")
    assert [p.name for p in discover(tmp_path)] == ["a.traj.json"]


def test_build_run_without_event_log(tmp_path):
    trajectory = write_trajectory(
        tmp_path,
        info={"exit_status": "Submitted", "model_stats": {"instance_cost": 0.25, "api_calls": 3}},
        messages=[{"role": "assistant"}, {"role": "assistant"}, {"role": "user"}],
    )
    run = build_run(trajectory)

    assert (run["outcome"], run["success"], run["cost"], run["n_calls"]) == ("Unverified", False, 0.25, 3)
    assert (run["wall_s"], run["llm_s"], run["env_s"], run["retry_count"]) == (None, None, None, None)
    assert (run["has_event_log"], run["n_steps"], run["n_steps_source"]) == (False, 2, "messages")


def test_build_run_with_event_log(tmp_path):
    trajectory = write_trajectory(tmp_path, info={"model_stats": {"instance_cost": 0.5, "api_calls": 1}})
    write_events(Path(f"{trajectory}.events.jsonl"), sample_events())

    run = build_run(trajectory)
    assert (run["wall_s"], run["llm_s"], run["env_s"], run["retry_count"], run["n_steps"]) == (4.0, 2.0, 0.5, 0, 1)
    assert (run["has_event_log"], run["exit_status"]) == (True, "Submitted")


def test_build_run_resolves_relative_event_path(tmp_path):
    trajectory = write_trajectory(tmp_path, info={"events": {"path": "custom.events.jsonl"}})
    write_events(tmp_path / "custom.events.jsonl", sample_events())
    assert build_run(trajectory)["has_event_log"] is True


def test_build_run_submitted_without_state_is_unverified(tmp_path):
    run = build_run(write_trajectory(tmp_path, info={"exit_status": "Submitted"}))
    assert (run["outcome"], run["success"]) == ("Unverified", False)


def test_build_run_corrupt_trajectory_is_error_record(tmp_path):
    path = tmp_path / "bad.traj.json"
    path.write_text("{not json")
    run = build_run(path)
    assert (run["outcome"], run["success"], run["error"] is not None) == ("Error", False, True)


def test_build_run_corrupt_event_log_keeps_metrics_none(tmp_path):
    trajectory = write_trajectory(tmp_path, info={"exit_status": "Submitted", "model_stats": {"instance_cost": 0.1}})
    Path(f"{trajectory}.events.jsonl").write_text('{"event": "a"}\nnot json\n{"event": "b"}\n')

    run = build_run(trajectory)
    assert (run["has_event_log"], run["event_log_error"] is not None) == (True, True)
    assert (run["wall_s"], run["retry_count"], run["cost"]) == (None, None, 0.1)


def test_build_run_accepts_legacy_message_list(tmp_path):
    path = tmp_path / "legacy.traj.json"
    path.write_text(json.dumps([{"role": "assistant"}, {"role": "user"}]))
    run = build_run(path)
    assert (run["outcome"], run["n_steps"], run["n_steps_source"]) == ("Incomplete", 1, "messages")


def test_grouping_prefers_instance_id_then_stem(tmp_path):
    grouped = write_trajectory(tmp_path, "grouped.traj.json", instance_id="task-1")
    plain = write_trajectory(tmp_path, "plain.traj.json")
    assert sorted(group_runs([build_run(grouped), build_run(plain)])) == ["plain", "task-1"]


def test_grouping_merges_repeated_runs(tmp_path):
    first = write_trajectory(tmp_path, "run1.traj.json", instance_id="task-1")
    second = write_trajectory(tmp_path, "run2.traj.json", instance_id="task-1")
    assert len(group_runs([build_run(first), build_run(second)])["task-1"]) == 2


def test_build_report_includes_summary_and_groups(tmp_path):
    write_trajectory(tmp_path, "ok.traj.json", info={"exit_status": "Submitted", "task_state": "Succeeded"})
    write_trajectory(tmp_path, "unknown.traj.json", info={"exit_status": "Submitted"})
    report = build_report(tmp_path)

    assert report["schema_version"] == 1
    assert (report["summary"]["n_runs"], report["summary"]["successes"]) == (2, 1)
    assert [g["group"] for g in report["groups"]] == ["ok", "unknown"]


def test_build_report_sorts_runs(tmp_path):
    write_trajectory(tmp_path, "b.traj.json", instance_id="z")
    write_trajectory(tmp_path, "a.traj.json", instance_id="a")
    runs = build_report(tmp_path)["runs"]
    assert [(r["group"], r["path"].rsplit("/", 1)[-1]) for r in runs] == [("a", "a.traj.json"), ("z", "b.traj.json")]


def test_pct_change_returns_none_for_zero_baseline():
    assert _pct_change(1.0, 0.0) is None
    assert _pct_change(1.0, 2.0) == -50.0


def test_compare_success_rate_uses_percentage_points():
    current = [make_run("Succeeded", group="a"), make_run("Succeeded", group="b")]
    baseline = [make_run("Succeeded", group="a"), make_run("Unverified", group="b")]
    assert compare(current, baseline)["overall"]["success_rate"]["delta_pp"] == 50.0


def test_compare_guards_zero_baseline():
    comparison = compare([make_run(cost=1.0)], [make_run(cost=0.0)])
    assert comparison["overall"]["cost_mean"]["delta"] == 1.0
    assert comparison["overall"]["cost_mean"]["delta_pct"] is None


def test_compare_reports_unmatched_groups():
    comparison = compare([make_run(group="a"), make_run(group="b")], [make_run(group="a"), make_run(group="c")])
    assert [m["group"] for m in comparison["matched"]] == ["a"]
    assert [g["group"] for g in comparison["only_current"]] == ["b"]
    assert [g["group"] for g in comparison["only_baseline"]] == ["c"]


def test_compare_reports_bucket_deltas():
    buckets = compare(
        [make_run("Succeeded"), make_run("Succeeded")],
        [make_run("Succeeded"), make_run("Unverified")],
    )["overall"]["buckets"]
    assert (buckets["Succeeded"]["delta"], buckets["Unverified"]["delta"]) == (1, -1)


def test_eval_cli_json_schema(tmp_path):
    write_trajectory(tmp_path, "ok.traj.json", info={"exit_status": "Submitted", "task_state": "Succeeded"})
    write_trajectory(tmp_path, "unknown.traj.json", info={"exit_status": "Submitted"})

    result = CliRunner().invoke(app, [str(tmp_path), "--json"])

    report = json.loads(result.output)
    assert set(report) == {"schema_version", "path", "runs", "summary", "groups", "baseline_path", "comparison"}
    assert (report["summary"]["n_runs"], report["summary"]["successes"]) == (2, 1)
    assert (report["baseline_path"], report["comparison"]) == (None, None)


def test_eval_cli_baseline_json(tmp_path):
    current, baseline = tmp_path / "current", tmp_path / "baseline"
    write_trajectory(current, "a.traj.json", info={"exit_status": "Submitted", "task_state": "Succeeded"})
    write_trajectory(baseline, "a.traj.json", info={"exit_status": "Submitted"})

    result = CliRunner().invoke(app, [str(current), "--baseline", str(baseline), "--json"])

    report = json.loads(result.output)
    assert report["comparison"]["overall"]["success_rate"]["delta_pp"] == 100.0
    assert report["comparison"]["matched"][0]["group"] == "a"


def test_eval_cli_verbose_prints_runs(tmp_path):
    write_trajectory(tmp_path, "uniq.traj.json", info={"exit_status": "Submitted"})
    plain = CliRunner().invoke(app, [str(tmp_path)])
    verbose = CliRunner().invoke(app, [str(tmp_path), "--verbose"])

    assert "uniq" not in plain.output
    assert "uniq" in verbose.output


def test_eval_cli_rejects_empty_directory(tmp_path):
    result = CliRunner().invoke(app, [str(tmp_path)])
    assert result.exit_code != 0
    assert "No trajectory files found" in result.output


def test_eval_cli_rejects_missing_path(tmp_path):
    missing = CliRunner().invoke(app, [str(tmp_path / "nope")])
    assert missing.exit_code != 0
    assert "does not exist" in missing.output

    baseline = CliRunner().invoke(app, [str(tmp_path), "--baseline", str(tmp_path / "nope")])
    assert baseline.exit_code != 0
    assert "Baseline path" in baseline.output


def test_eval_is_registered_in_mini_extra():
    assert any("eval" in aliases for _, aliases, _ in subcommands)
