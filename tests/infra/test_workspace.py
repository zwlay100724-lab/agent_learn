import subprocess
from pathlib import Path

from trace_swe.infra.workspace import workspace_fingerprint


def init_repo(path: Path) -> None:
    path.mkdir()
    (path / "a.py").write_text("one\n")
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=eval", "-c", "user.email=eval@example.com", "commit", "-qm", "initial"],
    ):
        subprocess.run(args, cwd=path, check=True, capture_output=True)


def test_fingerprint_detects_tracked_and_untracked_changes(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    before = workspace_fingerprint(repo)

    (repo / "a.py").write_text("two\n")
    after_tracked = workspace_fingerprint(repo)
    (repo / "b.py").write_text("new\n")
    after_untracked = workspace_fingerprint(repo)

    assert {before["kind"], after_tracked["kind"], after_untracked["kind"]} == {"git"}
    assert len({before["tree"], after_tracked["tree"], after_untracked["tree"]}) == 3
    assert (before["dirty"], after_tracked["dirty"]) == (False, True)
    assert (
        subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo, capture_output=True, text=True).stdout
        == ""
    )


def test_fingerprint_excludes_requested_paths(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    before = workspace_fingerprint(repo)

    (repo / "run.traj.json").write_text("{}")
    assert workspace_fingerprint(repo)["tree"] != before["tree"]
    assert workspace_fingerprint(repo, exclude=(repo / "run.traj.json",))["tree"] == before["tree"]


def test_fingerprint_reports_a_non_git_workspace(tmp_path):
    assert workspace_fingerprint(tmp_path)["kind"] == "unavailable"
