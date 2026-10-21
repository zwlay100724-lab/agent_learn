"""Offline evaluation of saved trajectories."""

import math
import statistics

OUTCOMES = ("Succeeded", "VerificationFailed", "Unverified", "Incomplete", "Error")
INCOMPLETE_EXITS = frozenset({"LimitsExceeded", "TimeExceeded", "RepeatedFormatError"})


def classify_outcome(task_state: str | None, exit_status: str | None) -> str:
    if task_state:
        return task_state if task_state in OUTCOMES else "Error"
    if exit_status is None or exit_status in INCOMPLETE_EXITS:
        return "Incomplete"
    if exit_status == "Submitted":
        return "Unverified"
    return "Error"


def _p95(values: list[float]) -> float | None:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None


def _stats(values: list[float | int | None]) -> dict:
    present = [v for v in values if v is not None]
    if not present:
        return {"total": None, "mean": None, "median": None, "p95": None, "n": 0}
    return {
        "total": sum(present),
        "mean": sum(present) / len(present),
        "median": statistics.median(present),
        "p95": _p95(present),
        "n": len(present),
    }


def aggregate(runs: list[dict]) -> dict:
    buckets = {outcome: 0 for outcome in OUTCOMES}
    for run in runs:
        buckets[run["outcome"]] += 1
    retries = [r["retry_count"] for r in runs if r.get("retry_count") is not None]
    return {
        "n_runs": len(runs),
        "successes": buckets["Succeeded"],
        "success_rate": buckets["Succeeded"] / len(runs) if runs else None,
        "buckets": buckets,
        "n_load_errors": sum(1 for r in runs if r.get("error")),
        "cost": _stats([r.get("cost") for r in runs]),
        "wall_s": _stats([r.get("wall_s") for r in runs]),
        "llm_s": _stats([r.get("llm_s") for r in runs]),
        "env_s": _stats([r.get("env_s") for r in runs]),
        "n_steps": _stats([r.get("n_steps") for r in runs]),
        "n_calls": _stats([r.get("n_calls") for r in runs]),
        "retries": {
            "total": sum(retries) if retries else None,
            "rate": sum(retries) / len(retries) if retries else None,
            "n": len(retries),
        },
        "coverage": {
            "task_state": sum(1 for r in runs if r.get("task_state") is not None),
            "event_log": sum(1 for r in runs if r.get("has_event_log")),
            "event_metrics": sum(1 for r in runs if r.get("wall_s") is not None),
            "event_log_errors": sum(1 for r in runs if r.get("event_log_error")),
            "verification": sum(1 for r in runs if r.get("verification_attempts") is not None),
            "delivery": sum(1 for r in runs if r.get("delivery_changed_files") is not None),
            "cost": sum(1 for r in runs if r.get("cost") is not None),
            "n_steps": sum(1 for r in runs if r.get("n_steps") is not None),
        },
    }