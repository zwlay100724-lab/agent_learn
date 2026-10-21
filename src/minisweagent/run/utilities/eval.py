"""Offline evaluation of saved trajectories."""

import json
import math
import statistics
from pathlib import Path

import typer

from minisweagent.run.utilities.events import summarize
from minisweagent.utils.event_log import read_events

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


def discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.traj.json"))
        if not files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
        return _dedupe(files)
    raise typer.BadParameter(f"Path '{path}' is neither a file nor a directory")


def _dedupe(paths: list[Path]) -> list[Path]:
    seen, result = set(), []
    for path in paths:
        try:
            key = path.resolve()
        except (OSError, RuntimeError):
            key = path.absolute()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _resolve_event_log(trajectory: Path, info: dict) -> Path | None:
    declared = (info.get("events") or {}).get("path")
    candidates = []
    if declared:
        declared_path = Path(declared)
        candidates.extend(
            [declared_path] if declared_path.is_absolute() else [trajectory.parent / declared_path, declared_path]
        )
    candidates.append(Path(f"{trajectory}.events.jsonl"))
    return next((p for p in candidates if p.is_file()), None)


def _stem(path: Path) -> str:
    if path.name.endswith(".traj.json"):
        return path.name[: -len(".traj.json")] or path.parent.name
    return path.stem or path.parent.name


def _empty_run(trajectory: Path) -> dict:
    return {
        "path": str(trajectory),
        "group": _stem(trajectory),
        "instance_id": None,
        "exit_status": None,
        "task_state": None,
        "outcome": "Error",
        "success": False,
        "cost": None,
        "n_calls": None,
        "n_steps": None,
        "n_steps_source": None,
        "wall_s": None,
        "llm_s": None,
        "env_s": None,
        "retry_count": None,
        "max_attempts": None,
        "has_event_log": False,
        "event_log_error": None,
        "verification_attempts": None,
        "verification_passed": None,
        "delivery_changed_files": None,
        "agent_type": None,
        "error": None,
    }


def build_run(trajectory: Path) -> dict:
    run = _empty_run(trajectory)
    try:
        data = json.loads(trajectory.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return {**run, "error": f"{type(e).__name__}: {e}"}
    if isinstance(data, list):
        data = {"messages": data}
    if not isinstance(data, dict):
        return {**run, "error": "Trajectory is neither an object nor a list"}

    info = data.get("info") or {}
    messages = data.get("messages") if isinstance(data.get("messages"), list) else None
    event_log = _resolve_event_log(trajectory, info)
    event_summary, event_error = {}, None
    if event_log is not None:
        try:
            event_summary = summarize(read_events(event_log))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            event_error = f"{type(e).__name__}: {e}"

    instance_id = data.get("instance_id") or info.get("instance_id")
    exit_status = info.get("exit_status") or event_summary.get("exit_status")
    task_state = info.get("task_state")
    outcome = classify_outcome(task_state, exit_status)
    model_stats = info.get("model_stats") or {}
    verification = info.get("verification") or {}
    delivery = info.get("delivery") or {}

    cost = model_stats.get("instance_cost")
    if cost is None:
        cost = event_summary.get("cost")
    n_steps, n_steps_source = None, None
    if event_summary.get("n_steps") is not None:
        n_steps, n_steps_source = event_summary["n_steps"], "events"
    elif messages is not None:
        n_steps, n_steps_source = sum(1 for m in messages if m.get("role") == "assistant"), "messages"

    return {
        **run,
        "group": instance_id or _stem(trajectory),
        "instance_id": instance_id,
        "exit_status": exit_status,
        "task_state": task_state,
        "outcome": outcome,
        "success": outcome == "Succeeded",
        "cost": cost,
        "n_calls": model_stats.get("api_calls"),
        "n_steps": n_steps,
        "n_steps_source": n_steps_source,
        "wall_s": event_summary.get("wall_s"),
        "llm_s": event_summary.get("llm_s"),
        "env_s": event_summary.get("env_s"),
        "retry_count": event_summary.get("retry_count"),
        "max_attempts": event_summary.get("max_attempts"),
        "has_event_log": event_log is not None,
        "event_log_error": event_error,
        "verification_attempts": verification.get("attempts"),
        "verification_passed": (verification.get("last") or {}).get("passed"),
        "delivery_changed_files": (
            len(delivery["changed_files"]) if isinstance(delivery.get("changed_files"), list) else None
        ),
        "agent_type": (info.get("config") or {}).get("agent_type"),
    }


def group_runs(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for run in sorted(runs, key=lambda r: (r["group"], r["path"])):
        groups.setdefault(run["group"], []).append(run)
    return groups


def _group_report(runs: list[dict]) -> list[dict]:
    return [
        {"group": group, "summary": aggregate(group_runs)} for group, group_runs in sorted(group_runs(runs).items())
    ]


def build_report(path: Path) -> dict:
    runs = sorted((build_run(p) for p in discover(path)), key=lambda r: (r["group"], r["path"]))
    return {
        "schema_version": 1,
        "path": str(path),
        "runs": runs,
        "summary": aggregate(runs),
        "groups": _group_report(runs),
    }


def _pct_change(current: float | int | None, baseline: float | int | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return (current - baseline) / baseline * 100


def _delta(current, baseline, *, pct: bool = False) -> dict:
    result = {"current": current, "baseline": baseline, "delta": None}
    if pct:
        result["delta_pct"] = None
    if current is not None and baseline is not None:
        result["delta"] = current - baseline
        if pct:
            result["delta_pct"] = _pct_change(current, baseline)
    return result


def _delta_summary(current: dict, baseline: dict) -> dict:
    return {
        "n_runs": _delta(current["n_runs"], baseline["n_runs"]),
        "successes": _delta(current["successes"], baseline["successes"]),
        "success_rate": {
            "current": current["success_rate"],
            "baseline": baseline["success_rate"],
            "delta_pp": (
                (current["success_rate"] - baseline["success_rate"]) * 100
                if current["success_rate"] is not None and baseline["success_rate"] is not None
                else None
            ),
        },
        "buckets": {
            outcome: _delta(current["buckets"][outcome], baseline["buckets"][outcome]) for outcome in OUTCOMES
        },
        "cost_total": _delta(current["cost"]["total"], baseline["cost"]["total"], pct=True),
        "cost_mean": _delta(current["cost"]["mean"], baseline["cost"]["mean"], pct=True),
        "wall_s_mean": _delta(current["wall_s"]["mean"], baseline["wall_s"]["mean"], pct=True),
        "n_steps_mean": _delta(current["n_steps"]["mean"], baseline["n_steps"]["mean"], pct=True),
        "retries_total": _delta(current["retries"]["total"], baseline["retries"]["total"], pct=True),
        "retry_rate": _delta(current["retries"]["rate"], baseline["retries"]["rate"], pct=True),
    }


def compare(current_runs: list[dict], baseline_runs: list[dict]) -> dict:
    current_groups, baseline_groups = group_runs(current_runs), group_runs(baseline_runs)
    return {
        "overall": _delta_summary(aggregate(current_runs), aggregate(baseline_runs)),
        "matched": [
            {"group": key, "delta": _delta_summary(aggregate(current_groups[key]), aggregate(baseline_groups[key]))}
            for key in sorted(current_groups.keys() & baseline_groups.keys())
        ],
        "only_current": [
            {"group": key, "summary": aggregate(current_groups[key])}
            for key in sorted(current_groups.keys() - baseline_groups.keys())
        ],
        "only_baseline": [
            {"group": key, "summary": aggregate(baseline_groups[key])}
            for key in sorted(baseline_groups.keys() - current_groups.keys())
        ],
    }