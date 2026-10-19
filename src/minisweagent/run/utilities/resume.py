#!/usr/bin/env python3

"""Resume an interrupted run from a saved trajectory."""

import json
from pathlib import Path

import typer
from rich.console import Console

from minisweagent.agents import get_agent_class
from minisweagent.agents.resume import ResumableAgent, ResumeBlocked
from minisweagent.environments import get_environment
from minisweagent.models import get_model

app = typer.Typer(rich_markup_mode="rich")
console = Console(highlight=False)


@app.command()
def main(
    trajectory: Path = typer.Argument(..., help="Trajectory (.traj.json) to resume"),
    policy: str = typer.Option("refuse", "--policy", help="refuse (default) or as_is"),
    check: str = typer.Option("", "--check", help="Workspace check command to record before resuming"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Where to save the continued trajectory"),
) -> None:
    if not trajectory.exists():
        raise typer.BadParameter(f"Trajectory '{trajectory}' does not exist")
    info = json.loads(trajectory.read_text())["info"]

    agent_config = dict(info["config"]["agent"])
    agent_config["output_path"] = output or agent_config.get("output_path") or trajectory
    agent_config["resume_policy"] = policy
    agent_config["resume_check_command"] = check

    model_config = dict(info["config"].get("model", {}))
    model_config["model_class"] = info["config"].get("model_type", "")
    env_config = {
        "environment_class": info["config"].get("environment_type", "local"),
        **info["config"].get("environment", {}),
    }
    agent_cls = ResumableAgent
    if agent_type := info["config"].get("agent_type"):
        try:
            candidate = get_agent_class(agent_type)
        except ValueError:
            candidate = None
        if candidate and issubclass(candidate, ResumableAgent):
            agent_cls = candidate
    agent = agent_cls(get_model(config=model_config), get_environment(env_config, default_type="local"), **agent_config)

    try:
        result = agent.resume(trajectory)
    except ResumeBlocked as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"Resumed run finished: [bold]{result.get('task_state') or result.get('exit_status')}[/bold]")


if __name__ == "__main__":
    app()
