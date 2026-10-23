"""Offline evaluation of saved trajectories."""

import json
import math
import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from trace_swe.cli.commands.events import summarize
from trace_swe.infra.event_log import read_events

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
        return {
            "total": None,
            "mean": None,
            "median": None,
            "p95": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0,
        }
    return {
        "total": sum(present),
        "mean": sum(present) / len(present),
        "median": statistics.median(present),
        "p95": _p95(present),
        "std": statistics.stdev(present) if len(present) > 1 else None,
        "min": min(present),
        "max": max(present),
        "n": len(present),
    }


def _failure_reason(run: dict) -> str | None:
    if run.get("outcome") == "Succeeded":
        return None
    if run.get("error"):
        return "load_error"
    if run.get("outcome") == "VerificationFailed":
        return "verification_failed"
    if run.get("exit_status") in ("LimitsExceeded", "TimeExceeded"):
        return "limits"
    if run.get("exit_status") == "RepeatedFormatError":
        return "format_error"
    if run.get("outcome") == "Unverified":
        return "unverified"
    if run.get("outcome") == "Incomplete":
        return "incomplete"
    return "other"


def aggregate(runs: list[dict]) -> dict:
    buckets = {outcome: 0 for outcome in OUTCOMES}
    for run in runs:
        buckets[run["outcome"]] += 1
    retries = [r["retry_count"] for r in runs if r.get("retry_count") is not None]
    prompt_calls = [
        r.get("calls_with_usage")
        for r in runs
        if r.get("prompt_tokens") is not None and r.get("calls_with_usage") is not None
    ]
    prompt_totals = [r.get("prompt_tokens") for r in runs if r.get("prompt_tokens") is not None]
    prompt_total = sum(prompt_totals) if prompt_totals else None
    prompt_call_count = sum(prompt_calls) if prompt_calls else 0
    failure_reasons: dict[str, int] = {}
    for run in runs:
        if reason := _failure_reason(run):
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
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
        "prompt_tokens": _stats(prompt_totals),
        "completion_tokens": _stats([r.get("completion_tokens") for r in runs]),
        "total_tokens": _stats([r.get("total_tokens") for r in runs]),
        "mean_prompt_tokens": _stats([r.get("mean_prompt_tokens") for r in runs]),
        "mean_prompt_per_call": prompt_total / prompt_call_count
        if prompt_total is not None and prompt_call_count
        else None,
        "elisions": _stats([r.get("elisions") for r in runs]),
        "elided_chars": _stats([r.get("elided_chars") for r in runs]),
        "failure_reasons": dict(sorted(failure_reasons.items())),
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
            "tokens": sum(1 for r in runs if r.get("prompt_tokens") is not None),
            "elisions": sum(1 for r in runs if r.get("elisions") is not None),
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
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "mean_prompt_tokens": None,
        "calls_with_usage": None,
        "elisions": None,
        "elided_chars": None,
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
    tokens = info.get("tokens") or event_summary.get("tokens") or {}
    context = info.get("context") or event_summary.get("context") or {}

    cost = model_stats.get("instance_cost")
    if cost is None:
        cost = event_summary.get("cost")
    n_steps, n_steps_source = None, None
    if event_summary.get("n_steps") is not None:
        n_steps, n_steps_source = event_summary["n_steps"], "events"
    elif messages is not None:
        n_steps, n_steps_source = sum(1 for m in messages if m.get("role") == "assistant"), "messages"
    calls_with_usage = tokens.get("n_calls_with_usage")
    prompt_tokens = tokens.get("prompt_tokens")

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
        "prompt_tokens": prompt_tokens,
        "completion_tokens": tokens.get("completion_tokens"),
        "total_tokens": tokens.get("total_tokens"),
        "mean_prompt_tokens": prompt_tokens / calls_with_usage
        if prompt_tokens is not None and calls_with_usage
        else None,
        "calls_with_usage": calls_with_usage,
        "elisions": context.get("elided_observations", context.get("elisions")),
        "elided_chars": context.get("elided_chars"),
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
        "buckets": {outcome: _delta(current["buckets"][outcome], baseline["buckets"][outcome]) for outcome in OUTCOMES},
        "cost_total": _delta(current["cost"]["total"], baseline["cost"]["total"], pct=True),
        "cost_mean": _delta(current["cost"]["mean"], baseline["cost"]["mean"], pct=True),
        "wall_s_mean": _delta(current["wall_s"]["mean"], baseline["wall_s"]["mean"], pct=True),
        "n_steps_mean": _delta(current["n_steps"]["mean"], baseline["n_steps"]["mean"], pct=True),
        "prompt_tokens_total": _delta(current["prompt_tokens"]["total"], baseline["prompt_tokens"]["total"], pct=True),
        "prompt_tokens_per_call": _delta(current["mean_prompt_per_call"], baseline["mean_prompt_per_call"], pct=True),
        "elisions_total": _delta(current["elisions"]["total"], baseline["elisions"]["total"], pct=True),
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


app = typer.Typer(rich_markup_mode="rich")
console = Console(highlight=False)


def _fmt(value, digits: int = 2, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def _fmt_int(value) -> str:
    return "n/a" if value is None else str(value)


def _fmt_money(value) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _render_summary(summary: dict) -> None:
    table = Table(header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("trajectories", str(summary["n_runs"]))
    table.add_row("Succeeded (strict)", f"{summary['successes']} ({_fmt_pct(summary['success_rate'])})")
    for outcome in OUTCOMES[1:]:
        table.add_row(outcome, str(summary["buckets"][outcome]))
    table.add_row(
        "cost total / mean / median",
        " / ".join(_fmt_money(summary["cost"][key]) for key in ("total", "mean", "median")),
    )
    table.add_row(
        "prompt tokens total / per call",
        f"{_fmt_int(summary['prompt_tokens']['total'])} / {_fmt(summary['mean_prompt_per_call'])}",
    )
    table.add_row("completion tokens total", _fmt_int(summary["completion_tokens"]["total"]))
    table.add_row("total tokens", _fmt_int(summary["total_tokens"]["total"]))
    table.add_row(
        "elisions / elided chars",
        f"{_fmt_int(summary['elisions']['total'])} / {_fmt_int(summary['elided_chars']['total'])}",
    )
    table.add_row("wall mean", _fmt(summary["wall_s"]["mean"], unit="s"))
    table.add_row("llm mean", _fmt(summary["llm_s"]["mean"], unit="s"))
    table.add_row("env mean", _fmt(summary["env_s"]["mean"], unit="s"))
    table.add_row("steps mean", _fmt(summary["n_steps"]["mean"]))
    table.add_row(
        "retries total / mean per run",
        f"{_fmt_int(summary['retries']['total'])} / {_fmt(summary['retries']['rate'])}",
    )
    console.print(table)


def _render_coverage(summary: dict) -> None:
    n, coverage = summary["n_runs"], summary["coverage"]
    if coverage["task_state"] < n:
        console.print(
            f"[yellow]{n - coverage['task_state']} run(s) have no task_state; "
            "they are not counted as successes.[/yellow]"
        )
    if coverage["event_log"] < n:
        console.print(
            f"[yellow]{n - coverage['event_log']} run(s) have no event log; "
            "timing/retry metrics are unavailable (not zero).[/yellow]"
        )
    if coverage["tokens"] < n:
        console.print(
            f"[yellow]{n - coverage['tokens']} run(s) have no token usage; "
            "token metrics are unavailable (not zero).[/yellow]"
        )
    if summary["n_load_errors"]:
        console.print(f"[red]{summary['n_load_errors']} trajectory file(s) could not be read; see --verbose.[/red]")


def _render_outcomes(summary: dict) -> None:
    table = Table(header_style="bold")
    table.add_column("outcome")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")
    for outcome in OUTCOMES:
        count = summary["buckets"][outcome]
        share = f"{count / summary['n_runs'] * 100:.1f}%" if summary["n_runs"] else "0.0%"
        table.add_row(outcome, str(count), share)
    console.print(table)


def _render_failures(summary: dict) -> None:
    if not summary["failure_reasons"]:
        return
    table = Table(title="Failure reasons", header_style="bold")
    table.add_column("reason")
    table.add_column("count", justify="right")
    for reason, count in summary["failure_reasons"].items():
        table.add_row(reason, str(count))
    console.print(table)


def _render_runs(report: dict) -> None:
    table = Table(header_style="bold")
    for column in ("group", "outcome", "cost", "calls", "steps", "wall", "retries", "prompt/call", "file"):
        table.add_column(column)
    for run in report["runs"]:
        table.add_row(
            run["group"],
            run["outcome"],
            _fmt_money(run["cost"]),
            _fmt_int(run["n_calls"]),
            _fmt_int(run["n_steps"]),
            _fmt(run["wall_s"], unit="s"),
            _fmt_int(run["retry_count"]),
            _fmt(run["mean_prompt_tokens"], 0),
            Path(run["path"]).name,
        )
    console.print(table)


def _render_comparison(comparison: dict) -> None:
    overall = comparison["overall"]
    table = Table(title="Comparison (current vs baseline)", header_style="bold")
    table.add_column("metric")
    table.add_column("current", justify="right")
    table.add_column("baseline", justify="right")
    table.add_column("delta", justify="right")
    success = overall["success_rate"]
    table.add_row(
        "strict success rate",
        _fmt_pct(success["current"]),
        _fmt_pct(success["baseline"]),
        f"{success['delta_pp']:+.1f}pp" if success["delta_pp"] is not None else "n/a",
    )
    for key, label, formatter in (
        ("cost_mean", "mean cost", _fmt_money),
        ("n_steps_mean", "mean steps", _fmt),
        ("wall_s_mean", "mean wall", lambda value: _fmt(value, unit="s")),
        ("retry_rate", "retry rate", _fmt),
        ("prompt_tokens_per_call", "prompt tokens/call", _fmt),
        ("elisions_total", "elisions", _fmt),
    ):
        delta = overall[key]
        if delta["delta"] is None:
            delta_text = "n/a"
        elif key == "cost_mean":
            delta_text = f"{delta['delta']:+.4f}"
        elif key in ("prompt_tokens_per_call", "elisions_total"):
            delta_text = f"{delta['delta']:+.0f}"
        else:
            delta_text = f"{delta['delta']:+.2f}"
        table.add_row(
            label,
            formatter(delta["current"]),
            formatter(delta["baseline"]),
            delta_text,
        )
    console.print(table)
    if comparison["matched"]:
        groups = Table(title="Groups", header_style="bold")
        for column in ("group", "runs cur/base", "success cur/base/delta", "cost delta %"):
            groups.add_column(column)
        for item in comparison["matched"]:
            delta = item["delta"]
            success_rate = delta["success_rate"]
            success_text = (
                f"{_fmt_pct(success_rate['current'])}/{_fmt_pct(success_rate['baseline'])}/"
                f"{success_rate['delta_pp']:+.1f}pp"
                if success_rate["delta_pp"] is not None
                else "n/a"
            )
            cost_pct = delta["cost_mean"]["delta_pct"]
            groups.add_row(
                item["group"],
                f"{delta['n_runs']['current']}/{delta['n_runs']['baseline']}",
                success_text,
                f"{cost_pct:+.1f}%" if cost_pct is not None else "n/a",
            )
        console.print(groups)
    for key, label in (("only_current", "only in current"), ("only_baseline", "only in baseline")):
        for item in comparison[key]:
            console.print(f"[yellow]{label}: {item['group']} ({item['summary']['n_runs']} runs)[/yellow]")


@app.command()
def main(
    path: Path = typer.Argument(..., help="Directory of .traj.json files or a single trajectory"),
    baseline: Path | None = typer.Option(None, "--baseline", help="Baseline directory or trajectory"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON instead of tables"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show one row per trajectory"),
) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Path '{path}' does not exist")
    if baseline is not None and not baseline.exists():
        raise typer.BadParameter(f"Baseline path '{baseline}' does not exist")

    report = build_report(path)
    if baseline is not None:
        report["baseline_path"] = str(baseline)
        report["comparison"] = compare(report["runs"], build_report(baseline)["runs"])
    else:
        report["baseline_path"] = None
        report["comparison"] = None

    if json_output:
        typer.echo(json.dumps(report, indent=2, allow_nan=False))
        return

    console.print(f"Evaluated [bold]{report['summary']['n_runs']}[/bold] trajectory file(s) from '{path}'")
    _render_summary(report["summary"])
    _render_outcomes(report["summary"])
    _render_failures(report["summary"])
    _render_coverage(report["summary"])
    if verbose:
        _render_runs(report)
    if report["comparison"]:
        _render_comparison(report["comparison"])


if __name__ == "__main__":
    app()
