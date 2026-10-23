"""Fingerprint a git workspace without touching its real index."""

import os
import subprocess
import tempfile
from pathlib import Path


def _relative(cwd: Path, paths: tuple[Path, ...]) -> list[str]:
    result = []
    for path in paths:
        try:
            result.append(path.resolve().relative_to(cwd.resolve()).as_posix())
        except (OSError, ValueError):
            continue
    return result


def workspace_fingerprint(cwd: Path, *, exclude: tuple[Path, ...] = (), timeout: int = 30) -> dict:
    """Return a content tree hash for a git worktree, including untracked files.

    The workspace's real index is not modified: all operations use a temporary index.
    """
    cwd = cwd.resolve()
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except OSError as e:
        return {"kind": "unavailable", "error": f"{type(e).__name__}: {e}"}
    if inside.returncode != 0:
        return {"kind": "unavailable", "error": "not a git worktree"}

    excludes = [f":(exclude){path}" for path in _relative(cwd, exclude)]
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ | {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        try:
            head = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            subprocess.run(
                ["git", "read-tree", "HEAD" if head.returncode == 0 else "--empty"],
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                timeout=timeout,
            )
            subprocess.run(
                ["git", "add", "-A", "--", ".", *excludes],
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                timeout=timeout,
            )
            tree = subprocess.run(
                ["git", "write-tree"],
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain", "--", ".", *excludes],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as e:
            return {"kind": "unavailable", "error": f"{type(e).__name__}: {e}"}
    return {
        "kind": "git",
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "tree": tree.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }
