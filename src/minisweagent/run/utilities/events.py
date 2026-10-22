#!/usr/bin/env python3

"""Summarize an event log: per-step timings, retries, cost, and failure attribution."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from minisweagent.models.utils.usage import sum_usage
from minisweagent.utils.event_log import read_events

app = typer.Typer(rich_markup_mode="rich")
console = Console(highlight=False)


def _context_summary(events: list[dict]) -> dict:
    elides = [e for e in events if e.get("event") == "context_elide"]
    return {
        "elisions": sum(e.get("count", 0) for e in elides),
        "elided_chars": sum(e.get("chars_elided", 0) for e in elides),
    }


def summarize(events: list[dict]) -> dict:
    """Aggregate the last run present in `events`."""
    if not events:
        return {"run_id": None, "incomplete": True, "steps": [], "failures": []}
    run = [e for e in events if e.get("run_id") == events[-1].get("run_id")]
    by_type = {
        t: [e for e in run if e.get("event") == t]
        for t in ("run_start", "run_end", "llm_call", "llm_retry", "step_end")
    }
    run_end = by_type["run_end"][0] if by_type["run_end"] else None
    run_start = by_type["run_start"][0] if by_type["run_start"] else run[0]
    return {
        "run_id": run[0].get("run_id"),
        "exit_status": (run_end or {}).get("exit_status")
        or next((e.get("exit_status") for e in reversed(by_type["step_end"]) if e.get("exit_status")), None),
        "task_state": (run_end or {}).get("task_state"),
        "incomplete": run_end is None,
        "wall_s": (run_end or {}).get("duration_s", run[-1].get("mono", 0.0) - run_start.get("mono", 0.0)),
        "llm_s": sum(e.get("duration_s", 0.0) for e in by_type["llm_call"]),
        "env_s": sum(e.get("duration_s", 0.0) for e in run if e.get("event") == "action"),
        "retry_count": len(by_type["llm_retry"]),
        "max_attempts": max((e.get("attempts", 1) for e in by_type["llm_call"]), default=0),
        "cost": sum(e.get("cost", 0.0) for e in by_type["llm_call"]),
        "tokens": sum_usage(by_type["llm_call"]),
        "context": _context_summary(run),
        "n_steps": len(by_type["step_end"]),
        "failures": [
            {"step": e.get("step"), "outcome": e.get("outcome"), "exit_status": e.get("exit_status")}
            for e in by_type["step_end"]
            if e.get("outcome") in ("format_error", "exception")
            or (e.get("outcome") == "interrupted" and e.get("exit_status") != "Submitted")
        ],
        "last_event": run[-1].get("event"),
        "steps": _step_rows(run),
    }


def _step_rows(events: list[dict]) -> list[dict]:
    rows: dict[int, dict] = {}
    for e in events:
        if (step := e.get("step", 0)) <= 0:
            continue
        row = rows.setdefault(
            step,
            {
                "step": step,
                "llm_s": 0.0,
                "calls": 0,
                "retries": 0,
                "env_s": 0.0,
                "actions": 0,
                "cost": 0.0,
                "outcome": "",
            },
        )
        if (event := e.get("event")) == "llm_call":
            row["llm_s"] += e.get("duration_s", 0.0)
            row["calls"] += 1
            row["cost"] += e.get("cost", 0.0)
        elif event == "llm_retry":
            row["retries"] += 1
        elif event == "action":
            row["env_s"] += e.get("duration_s", 0.0)
            row["actions"] += 1
        elif event == "step_end":
            row["outcome"] = e.get("outcome", "")
    return [rows[k] for k in sorted(rows)]


def _resolve(path: Path) -> Path:
    if path.suffix == ".jsonl":
        return path
    events = (json.loads(path.read_text()).get("info", {}).get("events") or {}).get("path")
    candidate = Path(events) if events else Path(f"{path}.events.jsonl")
    if not candidate.exists():
        raise typer.BadParameter(f"No event log found at '{candidate}'")
    return candidate


@app.command()
def main(path: Path = typer.Argument(..., help="Event log (.jsonl) or trajectory (.traj.json)")) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Path '{path}' does not exist")
    summary = summarize(read_events(_resolve(path)))
    if not summary["run_id"]:
        raise typer.BadParameter(f"Event log for '{path}' is empty")

    status = "INCOMPLETE" if summary["incomplete"] else (summary["task_state"] or summary["exit_status"])
    console.print(
        f"Run [bold]{summary['run_id']}[/bold]: {status}  wall={summary['wall_s']:.2f}s  "
        f"llm={summary['llm_s']:.2f}s  env={summary['env_s']:.2f}s  "
        f"retries={summary['retry_count']}  cost=${summary['cost']:.4f}"
    )
    table = Table(header_style="bold")
    for column in ("step", "llm_s", "calls", "retries", "env_s", "actions", "cost", "outcome"):
        table.add_column(column, justify="left" if column == "outcome" else "right")
    for row in summary["steps"]:
        table.add_row(
            str(row["step"]),
            f"{row['llm_s']:.2f}",
            str(row["calls"]),
            str(row["retries"]),
            f"{row['env_s']:.2f}",
            str(row["actions"]),
            f"${row['cost']:.4f}",
            row["outcome"],
        )
    console.print(table)
    if summary["incomplete"]:
        console.print(f"[yellow]No run_end event; last event was '{summary['last_event']}'[/yellow]")
    for failure in summary["failures"]:
        console.print(f"[red]step {failure['step']}: {failure['outcome']} {failure['exit_status'] or ''}[/red]")


if __name__ == "__main__":
    app()
