"""Run the context-management experiment and write a reproducible report.

Example:
    .venv/bin/python experiments/context_tasks/run_experiment.py --runs 5

Results go to experiments/context_tasks/results (override with --output).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
VENV_BIN = PROJECT / ".venv" / "bin"
MINI, MINI_EXTRA = VENV_BIN / "mini", VENV_BIN / "mini-extra"
DEFAULT_OUT = ROOT / "results"
DEFAULT_TASKS = "sum_range,log_errors"
DEFAULT_VARIANTS = "event_log,context"
PROTECTED = ("test_buggy.py", "app.log")
VARIANTS = {
    "event_log": [],
    "context": ["-c", "agent.context_max_observations=2", "-c", "agent.context_elide_min_chars=200"],
}

app = typer.Typer(add_completion=False)
console = Console()


def prepare_repo(task: str, work_dir: Path) -> None:
    shutil.copytree(
        ROOT / "tasks" / task / "template", work_dir, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=eval", "-c", "user.email=eval@example.com", "commit", "-qm", "initial"],
    ):
        subprocess.run(args, cwd=work_dir, check=True, capture_output=True)


def score(repo: Path, trajectory: Path) -> bool:
    changed = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", *PROTECTED], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    )
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo, capture_output=True, text=True)
    passed = result.returncode == 0 and not changed
    data = json.loads(trajectory.read_text())
    info = data.setdefault("info", {})
    info["task_state"] = "Succeeded" if passed else "VerificationFailed"
    info["verification"] = {
        "command": "python -m pytest -q",
        "attempts": 1,
        "last": {"passed": passed, "returncode": result.returncode, "tests_modified": changed},
    }
    trajectory.write_text(json.dumps(data, indent=2))
    return passed


def write_missing_trajectory(path: Path, reason: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "info": {
            "exit_status": "NoTrajectory",
            "task_state": "Error",
            "model_stats": {"instance_cost": 0.0, "api_calls": 0},
            "verification": {"command": "python -m pytest -q", "attempts": 0, "last": {"passed": False}},
            "error": f"{reason}: {detail[-1000:]}",
        },
        "messages": [],
    }
    path.write_text(json.dumps(data, indent=2))


def task_set_hash(tasks: list[str]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        for path in sorted((ROOT / "tasks" / task).rglob("*")):
            if path.is_file() and not {"__pycache__", ".pytest_cache"} & set(path.parts):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT, capture_output=True, text=True).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def run_once(
    variant: str,
    task: str,
    run_index: int,
    output: Path,
    *,
    model: str,
    temperature: float,
    step_limit: int,
    cost_limit: float,
    timeout: int,
) -> dict:
    run_dir = output / variant / task / f"run-{run_index:02d}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    repo = run_dir / "repo"
    prepare_repo(task, repo)
    trajectory = run_dir / f"{task}.traj.json"
    command = [
        str(MINI),
        "--agent-class",
        variant,
        "-m",
        model,
        "-y",
        "--exit-immediately",
        "-c",
        "trace.yaml",
        "-c",
        f"agent.step_limit={step_limit}",
        "-c",
        f"model.model_kwargs.temperature={temperature}",
        "-c",
        f"environment.cwd={repo}",
        "-l",
        str(cost_limit),
        "-o",
        str(trajectory),
        "-t",
        (ROOT / "tasks" / task / "task.txt").read_text().strip(),
        *VARIANTS[variant],
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=os.environ | {"TRACE_SWE_SILENT_STARTUP": "1"},
        )
    except subprocess.TimeoutExpired as e:
        (run_dir / "mini.log").write_text(str(e))
        write_missing_trajectory(trajectory, "timeout", str(e))
        return {"variant": variant, "task": task, "run": run_index, "status": "timeout", "seconds": timeout}
    (run_dir / "mini.log").write_text(result.stdout + "\n" + result.stderr)
    if not trajectory.exists():
        write_missing_trajectory(trajectory, f"mini exited with {result.returncode}", result.stderr)
        status = "no_trajectory"
    else:
        status = "ok" if score(repo, trajectory) else "failed"
    return {"variant": variant, "task": task, "run": run_index, "status": status, "seconds": time.monotonic() - started}


@app.command()
def main(
    runs: int = typer.Option(5, "--runs", min=1, help="Runs per task and variant"),
    tasks: str = typer.Option(DEFAULT_TASKS, "--tasks", help="Comma-separated task names"),
    variants: str = typer.Option(DEFAULT_VARIANTS, "--variants", help="Comma-separated variants"),
    model: str = typer.Option(os.getenv("EVAL_MODEL", "deepseek/deepseek-flash"), "--model"),
    output: Path = typer.Option(DEFAULT_OUT, "-o", "--output"),
    temperature: float = typer.Option(0.0, "--temperature"),
    step_limit: int = typer.Option(12, "--step-limit"),
    cost_limit: float = typer.Option(0.05, "--cost-limit"),
    timeout: int = typer.Option(300, "--timeout"),
) -> None:
    task_names = [t.strip() for t in tasks.split(",") if t.strip()]
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]
    if unknown := set(variant_names) - set(VARIANTS):
        raise typer.BadParameter(f"Unknown variants: {sorted(unknown)}")
    if unknown_tasks := set(task_names) - {p.name for p in (ROOT / "tasks").iterdir() if p.is_dir()}:
        raise typer.BadParameter(f"Unknown tasks: {sorted(unknown_tasks)}")

    output = output.resolve()
    for variant in variant_names:
        shutil.rmtree(output / variant, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "step_limit": step_limit,
        "cost_limit": cost_limit,
        "runs": runs,
        "tasks": task_names,
        "variants": variant_names,
        "task_set_hash": task_set_hash(task_names),
        "git": git_state(),
        "python": sys.version.split()[0],
        "command": [sys.executable, *sys.argv],
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2))

    for variant in variant_names:
        for task in task_names:
            for run_index in range(1, runs + 1):
                result = run_once(
                    variant,
                    task,
                    run_index,
                    output,
                    model=model,
                    temperature=temperature,
                    step_limit=step_limit,
                    cost_limit=cost_limit,
                    timeout=timeout,
                )
                console.print(
                    f"[bold]{variant:9s}[/bold] {task:11s} run {run_index}/{runs} "
                    f"{result['status']:13s} {result['seconds']:.1f}s"
                )

    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output / "meta.json").write_text(json.dumps(meta, indent=2))

    if not {"event_log", "context"} <= set(variant_names):
        console.print("[yellow]Need both event_log and context variants for a baseline comparison.[/yellow]")
        return
    current, baseline = output / "context", output / "event_log"
    eval_command = [str(MINI_EXTRA), "eval", str(current), "--baseline", str(baseline)]
    result = subprocess.run(
        [*eval_command, "--json"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        env=os.environ | {"TRACE_SWE_SILENT_STARTUP": "1"},
        check=True,
    )
    report = json.loads(result.stdout)
    report["experiment"] = meta
    (output / "report.json").write_text(json.dumps(report, indent=2))
    subprocess.run(eval_command, cwd=PROJECT, env=os.environ | {"TRACE_SWE_SILENT_STARTUP": "1"}, check=True)
    console.print(f"Report written to [bold]{output / 'report.json'}[/bold]")


if __name__ == "__main__":
    app()
