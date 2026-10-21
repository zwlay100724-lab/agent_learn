import pytest

from minisweagent.run.utilities.eval import OUTCOMES, _p95, _stats, aggregate, classify_outcome


def make_run(outcome="Succeeded", **kwargs):
    run = {
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