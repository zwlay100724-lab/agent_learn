"""Resume an interrupted run from its saved trajectory.

The trajectory holds the conversation; the event log tells us where the process died.
Resuming is refused by default when the last event is not a snapshot, because the in-flight
step may already have changed the workspace.
"""

import json
from pathlib import Path
from typing import Literal

from minisweagent.agents.event_log import EventLogAgent, EventLogAgentConfig
from minisweagent.utils.event_log import read_events


class ResumeBlocked(RuntimeError):
    """Raised when the saved state cannot prove that resuming is safe."""


class ResumeConfig(EventLogAgentConfig):
    resume_policy: Literal["refuse", "as_is"] = "refuse"
    """refuse: stop on a non-snapshot boundary. as_is: continue anyway (at-least-once)."""
    resume_check_command: str = ""
    """Optional command run before resuming; its output is recorded as a workspace_probe event."""


class ResumableAgent(EventLogAgent):
    def __init__(self, *args, config_class: type = ResumeConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def resume(self, trajectory_path: Path) -> dict:
        data = json.loads(trajectory_path.read_text())
        info = data.get("info", {})
        self.messages = data["messages"]
        self.cost = info.get("model_stats", {}).get("instance_cost", 0.0)
        self.n_calls = info.get("model_stats", {}).get("api_calls", 0)
        self._task = info.get("task", "")
        self.extra_template_vars |= {"task": self._task}

        state = self._resume_state(info)
        self._check_can_resume(state)
        self.events.new_run(state["run_id"], start_seq=state["next_seq"])
        self._step_index = state["steps"]
        probe = self._probe_workspace() if self.config.resume_check_command else None
        return self._run_with_events(
            self._run_loop,
            "run_resumed",
            task=self._task,
            trajectory=str(trajectory_path),
            clean=state["clean"],
            last_event=state["last_event"],
            inflight=[_summarize_event(e) for e in state["inflight"]],
            workspace_probe=probe,
        )

    def _resume_state(self, info: dict) -> dict:
        events_config = info.get("events") or {}
        events = []
        if log_path := events_config.get("path"):
            if Path(log_path).exists():
                events = read_events(Path(log_path))
                if run_id := events_config.get("run_id"):
                    events = [e for e in events if e.get("run_id") == run_id] or events
        last_snapshot = max((i for i, e in enumerate(events) if e.get("event") == "snapshot_written"), default=-1)
        return {
            "run_id": events_config.get("run_id") or (events[-1].get("run_id") if events else None),
            "next_seq": events[-1].get("seq", 0) + 1 if events else 0,
            "steps": sum(1 for e in events if e.get("event") == "step_end"),
            "last_event": events[-1].get("event") if events else None,
            "clean": (events[-1].get("event") in ("snapshot_written", "run_end")) if events else None,
            "inflight": events[last_snapshot + 1 :],
        }

    def _check_can_resume(self, state: dict) -> None:
        if self.messages and self.messages[-1].get("role") == "exit":
            raise ResumeBlocked("The trajectory ends with an exit message; the run already finished.")
        if state["last_event"] == "run_end":
            raise ResumeBlocked(f"Run '{state['run_id']}' already finished; there is nothing to resume.")
        if state["clean"] is not False or self.config.resume_policy == "as_is":
            return
        commands = [e.get("command_preview", "") for e in state["inflight"] if e.get("event") == "action"]
        actions = "\n".join(f"  - {c}" for c in commands) or "  (none recorded)"
        raise ResumeBlocked(
            f"Refusing to resume: the last recorded event is '{state['last_event']}', not a snapshot, so the "
            f"in-flight step is missing from the trajectory.\nActions recorded after the last snapshot:\n{actions}\n"
            "These may already have taken effect. Inspect the workspace, then retry with resume_policy='as_is'."
        )

    def _probe_workspace(self) -> dict:
        output = self.env.probe(self.config.resume_check_command)
        return {"returncode": output.get("returncode"), "output": (output.get("output") or "")[:2000]}


def _summarize_event(event: dict) -> dict:
    return {key: event[key] for key in ("event", "step", "command_preview", "submitted") if key in event}
