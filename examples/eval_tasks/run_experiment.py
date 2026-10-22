"""Run the same tasks with and without context management, then compare.

Usage: .venv/bin/python examples/eval_tasks/run_experiment.py
Results go to /tmp/agent_learn_experiment (override with EVAL_OUT).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
VENV_BIN = PROJECT / ".venv" / "bin"
MINI, MINI_EXTRA = VENV_BIN / "mini", VENV_BIN / "mini-extra"
OUT = Path(os.getenv("EVAL_OUT", "/tmp/agent_learn_experiment"))
TASKS = ("sum_range", "parse_kv", "dedupe")
VARIANTS = {
    "event_log": [],
    "context": ["-c", "agent.context_max_observations=2", "-c", "agent.context_elide_min_chars=200"],
}


def prepare_repo(task: str, work_dir: Path) -> None:
    shutil.copytree(ROOT / "tasks" / task / "template", work_dir)
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=eval", "-c", "user.email=eval@example.com", "commit", "-qm", "initial"],
    ):
        subprocess.run(args, cwd=work_dir, check=True)


def score(repo: Path, trajectory: Path) -> bool:
    tests_changed = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "test_buggy.py"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    )
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo, capture_output=True, text=True)
    passed = result.returncode == 0 and not tests_changed
    data = json.loads(trajectory.read_text())
    data.setdefault("info", {})["task_state"] = "Succeeded" if passed else "VerificationFailed"
    data["info"]["verification"] = {
        "command": "python -m pytest -q",
        "attempts": 1,
        "last": {"passed": passed, "returncode": result.returncode, "tests_modified": tests_changed},
    }
    trajectory.write_text(json.dumps(data, indent=2))
    return passed


def run_variant(variant: str, task: str) -> None:
    out = OUT / variant / task
    shutil.rmtree(out, ignore_errors=True)
    repo, out = out / "repo", out
    out.mkdir(parents=True)
    prepare_repo(task, repo)
    trajectory = out / f"{task}.traj.json"
    env = os.environ | {"PATH": f"{VENV_BIN}:{os.environ['PATH']}"}
    command = [
        str(MINI),
        "--agent-class",
        variant,
        "-y",
        "-c",
        "mini.yaml",
        "-c",
        "agent.step_limit=12",
        "-c",
        "model.model_kwargs.temperature=0",
        "-c",
        f"environment.cwd={repo}",
        "-l",
        "0.05",
        "-o",
        str(trajectory),
        "-t",
        (ROOT / "tasks" / task / "task.txt").read_text().strip(),
        *VARIANTS[variant],
    ]
    subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    if not trajectory.exists():
        print(f"{variant:10s} {task:10s} no trajectory", flush=True)
        return
    passed = score(repo, trajectory)
    context = json.loads(trajectory.read_text()).get("info", {}).get("context") or {}
    print(
        f"{variant:10s} {task:10s} pytest={'pass' if passed else 'fail'} "
        f"elided={context.get('elided_observations', 0)}",
        flush=True,
    )


def main() -> None:
    for variant in VARIANTS:
        for task in TASKS:
            run_variant(variant, task)
    result = subprocess.run(
        [str(MINI_EXTRA), "eval", str(OUT / "context"), "--baseline", str(OUT / "event_log")],
        capture_output=True,
        text=True,
    )
    print(result.stdout, flush=True)


if __name__ == "__main__":
    main()