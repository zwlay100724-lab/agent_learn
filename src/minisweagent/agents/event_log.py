"""Agent that records process facts to an append-only event log.

The log is a side-channel: messages and the saved trajectory are unchanged. It captures
what the trajectory cannot - per-step timing, retry attempts, action outcomes, and output
size/hash - without modifying DefaultAgent or any model/environment implementation.
"""

import time
from pathlib import Path
from typing import Any

from minisweagent import __version__
from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
from minisweagent.exceptions import FormatError, InterruptAgentFlow, Submitted
from minisweagent.models.utils.usage import USAGE_FIELDS, extract_usage, sum_usage
from minisweagent.utils.event_log import (
    SCHEMA_VERSION,
    EventLogWriter,
    bind_event_log,
    digest,
    text_digest,
)
from minisweagent.utils.output_store import OutputStore
from minisweagent.utils.workspace import workspace_fingerprint


def _exit_status(e: BaseException) -> str | None:
    messages = getattr(e, "messages", ())
    return messages[0].get("extra", {}).get("exit_status") if messages else None


def _type_name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__name__}"


def _usage_payload(usage: dict | None) -> dict:
    return {key: (usage or {}).get(key) for key in USAGE_FIELDS}


def _message_usage(message: dict | None) -> dict | None:
    """Completion models keep the raw response in extra; response API models put usage at the top level."""
    if message is None:
        return None
    return extract_usage((message.get("extra") or {}).get("response")) or extract_usage(message)


def _record_usage(message: dict | None, usage: dict | None) -> None:
    if message is None or usage is None:
        return
    if message.get("extra") is None:
        message["extra"] = {}
    message["extra"]["usage"] = usage


def _token_totals(messages: list[dict]) -> dict:
    return sum_usage([usage for message in messages if (usage := (message.get("extra") or {}).get("usage"))])


class EventLogAgentConfig(InteractiveAgentConfig):
    event_log_path: Path | None = None
    """Where to write events. Defaults to `<output_path>.events.jsonl`."""
    event_log_output_preview_chars: int = 500
    """How many characters of a command's output to keep in the log."""
    event_log_command_preview_chars: int = 2000
    """How many characters of a command to keep in the log."""
    event_log_run_id: str | None = None
    """Reuse a run id (e.g. when resuming); a new one is generated otherwise."""
    output_offload_enabled: bool = True
    """Store large command outputs on disk. Only applies to LocalEnvironment."""
    output_offload_threshold_chars: int = 10_000
    """Outputs longer than this are written to disk and referenced in the observation."""
    output_offload_max_bytes: int = 5_000_000
    """Hard cap for a single stored output."""
    output_offload_dir: Path | None = None
    """Where to store outputs. Defaults to `<event log path>.outputs`."""
    workspace_fingerprint_timeout_seconds: int = 30
    """Timeout for each git command used to fingerprint the workspace."""


class _EventLogModel:
    """Transparent proxy that times each real model call and forwards everything else."""

    def __init__(self, model, writer: EventLogWriter):
        self._model = model
        self._writer = writer
        self.call_index = 0

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.call_index += 1
        retries_before = self._writer.retry_count
        started = time.monotonic()
        payload: dict[str, Any] = {"call_index": self.call_index}
        try:
            with bind_event_log(self._writer):
                message = self._model.query(messages, **kwargs)
        except FormatError as e:
            usage = _message_usage(e.messages[0] if e.messages else None)
            _record_usage(e.messages[0] if e.messages else None, usage)
            payload |= {
                "cost": e.messages[0].get("extra", {}).get("cost", 0.0) if e.messages else 0.0,
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "message_index": len(messages),
                "message_digest": digest(e.messages[0]) if e.messages else None,
                **_usage_payload(usage),
            }
            raise
        except Exception as e:
            payload |= {
                "cost": 0.0,
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                **_usage_payload(None),
            }
            raise
        else:
            usage = _message_usage(message)
            _record_usage(message, usage)
            payload |= {
                "cost": message.get("extra", {}).get("cost", 0.0),
                "success": True,
                "message_index": len(messages),
                "message_digest": digest(message),
                "actions_count": len(message.get("extra", {}).get("actions", [])),
                **_usage_payload(usage),
            }
            return message
        finally:
            payload["duration_s"] = round(time.monotonic() - started, 6)
            payload["attempts"] = 1 + (self._writer.retry_count - retries_before)
            self._writer.emit("llm_call", **payload)

    def __getattr__(self, name):
        return getattr(self._model, name)


class _EventLogEnvironment:
    """Transparent proxy that times action execution and forwards everything else."""

    def __init__(
        self,
        env,
        writer: EventLogWriter,
        output_preview_chars: int,
        command_preview_chars: int,
        output_store: OutputStore | None = None,
        offload_threshold_chars: int = 10_000,
    ):
        self._env = env
        self._writer = writer
        self._output_preview_chars = output_preview_chars
        self._command_preview_chars = command_preview_chars
        self._output_store = output_store
        self._offload_threshold_chars = offload_threshold_chars
        self.action_index = 0

    def _output_payload(self, raw: str) -> dict:
        return {
            "output_chars": len(raw),
            "output_bytes": len(raw.encode("utf-8", errors="replace")),
            "output_sha256": text_digest(raw),
            "output_preview": raw[: self._output_preview_chars],
            "output_truncated": len(raw) > self._output_preview_chars,
        }

    def _maybe_offload(self, output: dict | None, raw: str, *, label: str) -> str | None:
        if (
            output is None
            or self._output_store is None
            or not self._writer.enabled
            or len(raw) < self._offload_threshold_chars
        ):
            return None
        name = f"{self._writer.run_id}/{self._writer.seq:06d}-{label}.txt"
        if ref := self._output_store.save(raw, name):
            output["output_path"] = str(ref)
            return str(ref)
        return None

    def execute(self, action: dict, *args, **kwargs) -> dict:
        self.action_index += 1
        started = time.monotonic()
        output = None
        error = None
        try:
            output = self._env.execute(action, *args, **kwargs)
            return output
        except Submitted as e:
            error = e
            raise
        except Exception as e:
            error = e
            raise
        finally:
            command = action.get("command", "")
            raw = (output or {}).get("output", "") or ""
            output_ref = self._maybe_offload(output, raw, label=f"step{self._writer.step}")
            self._writer.emit(
                "action",
                action_index=self.action_index,
                command_preview=command[: self._command_preview_chars],
                command_chars=len(command),
                duration_s=round(time.monotonic() - started, 6),
                success=error is None or isinstance(error, Submitted),
                submitted=isinstance(error, Submitted),
                returncode=(output or {}).get("returncode"),
                exception_type=type(error).__name__ if error else None,
                exception_message=str(error) or None,
                output_ref=output_ref,
                **self._output_payload(raw),
            )

    def check(
        self,
        command: str,
        *,
        event: str = "verification",
        offload: bool = True,
        timeout: int | None = None,
        **fields,
    ) -> dict:
        """Run a non-agent command (verification, delivery) and never let it end the run."""
        started = time.monotonic()
        output, error = None, None
        try:
            kwargs = {"timeout": timeout} if timeout is not None else {}
            output = self._env.execute({"command": command}, **kwargs)
        except Submitted as e:
            # A verifier that prints the completion marker must not end the agent loop
            submission = e.messages[0].get("content", "") if e.messages else ""
            output = {"output": submission, "returncode": 0, "exception_info": ""}
        except Exception as e:
            error = e
            output = {"output": getattr(e, "output", "") or "", "returncode": -1, "exception_info": str(e)}
        finally:
            raw = (output or {}).get("output", "") or ""
            output_ref = self._maybe_offload(output, raw, label=f"{event}-step{self._writer.step}") if offload else None
            returncode = (output or {}).get("returncode")
            passed = (returncode == 0) if error is None else None
            output["passed"] = passed
            output["output_ref"] = output_ref
            output["output_preview"] = raw[: self._output_preview_chars]
            self._writer.emit(
                event,
                command=command,
                passed=passed,
                returncode=returncode,
                duration_s=round(time.monotonic() - started, 6),
                exception_type=type(error).__name__ if error else None,
                exception_message=(str(error) or None) if error else None,
                output_ref=output_ref,
                **self._output_payload(raw),
                **fields,
            )
        return output

    def probe(self, command: str) -> dict:
        """Run a workspace check without recording it as an agent action."""
        output = self._env.execute({"command": command})
        raw = output.get("output", "") or ""
        self._writer.emit(
            "workspace_probe",
            command=command,
            returncode=output.get("returncode"),
            output_preview=raw[: self._output_preview_chars],
            output_truncated=len(raw) > self._output_preview_chars,
            output_sha256=text_digest(raw),
        )
        return output

    def __getattr__(self, name):
        return getattr(self._env, name)


class EventLogAgent(InteractiveAgent):
    """InteractiveAgent that additionally writes a JSONL event log."""

    def __init__(self, model, env, *, config_class: type = EventLogAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._model_type, self._env_type = _type_name(model), _type_name(env)
        self.events = EventLogWriter(self._event_log_path(), run_id=self.config.event_log_run_id)
        self.model = _EventLogModel(self.model, self.events)
        self.env = _EventLogEnvironment(
            self.env,
            self.events,
            self.config.event_log_output_preview_chars,
            self.config.event_log_command_preview_chars,
            self._output_store(),
            self.config.output_offload_threshold_chars,
        )
        self._step_index = 0
        self._task = ""

    def _event_log_path(self) -> Path | None:
        if self.config.event_log_path:
            return Path(self.config.event_log_path)
        return Path(f"{self.config.output_path}.events.jsonl") if self.config.output_path else None

    def _output_store(self) -> OutputStore | None:
        if not self.config.output_offload_enabled or not self._env_type.endswith(".LocalEnvironment"):
            return None
        if self.config.output_offload_dir:
            root = Path(self.config.output_offload_dir)
        elif self.events.path:
            root = Path(f"{self.events.path}.outputs")
        else:
            return None
        return OutputStore(root, self.config.output_offload_max_bytes)

    def _run_end_extra(self, last_extra: dict) -> dict:
        return {}

    def _run_end_success(self, last_extra: dict, exit_status: str | None) -> bool:
        return exit_status == "Submitted"

    def run(self, task: str = "", **kwargs) -> dict:
        self._task = task
        self._step_index = 0
        self.events.new_run(self.config.event_log_run_id)
        return self._run_with_events(lambda: super(EventLogAgent, self).run(task, **kwargs), "run_start", task=task)

    def _run_with_events(self, start, event: str, **payload) -> dict:
        """Run `start` while bracketing it with a start event and a guaranteed run_end."""
        self.events.emit(
            event,
            **payload,
            agent_type=_type_name(self),
            model_type=self._model_type,
            env_type=self._env_type,
            output_path=str(self.config.output_path) if self.config.output_path else None,
            event_log_path=str(self.events.path) if self.events.path else None,
            mini_version=__version__,
        )
        started = time.monotonic()
        error = None
        try:
            return start()
        except Exception as e:
            error = e
            raise
        finally:
            last_extra = self.messages[-1].get("extra", {}) if self.messages else {}
            exit_status = last_extra.get("exit_status")
            self.events.emit(
                "run_end",
                duration_s=round(time.monotonic() - started, 6),
                exit_status=exit_status,
                submission_chars=len(last_extra.get("submission") or ""),
                cost=self.cost,
                n_calls=self.n_calls,
                n_steps=self._step_index,
                success=self._run_end_success(last_extra, exit_status),
                **self._run_end_extra(last_extra),
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
            )
            self.events.close()

    def step(self) -> list[dict]:
        self._step_index += 1
        self.events.step = self._step_index
        self.events.emit("step_start", n_calls=self.n_calls, n_messages=len(self.messages))
        started = time.monotonic()
        outcome, exit_status = "ok", None
        try:
            return super().step()
        except FormatError:
            outcome = "format_error"
            raise
        except InterruptAgentFlow as e:
            outcome, exit_status = "interrupted", _exit_status(e)
            raise
        except Exception:
            outcome = "exception"
            raise
        finally:
            self.events.emit(
                "step_end",
                duration_s=round(time.monotonic() - started, 6),
                outcome=outcome,
                exit_status=exit_status,
            )

    def _workspace_exclusions(self, output_path: Path | None) -> tuple[Path, ...]:
        paths = []
        if output_path:
            paths.extend(
                [Path(output_path), Path(f"{output_path}.patch.diff"), Path(f"{output_path}.verification.json")]
            )
        if self.events.path:
            paths.extend([self.events.path, Path(f"{self.events.path}.outputs"), Path(f"{self.events.path}.context")])
        return tuple(paths)

    def _fingerprint_workspace(self, output_path: Path | None) -> dict:
        if not self._env_type.endswith(".LocalEnvironment"):
            return {"kind": "remote"}
        config = getattr(self.env, "config", None)
        cwd = Path(getattr(config, "cwd", "") or Path.cwd())
        return workspace_fingerprint(
            cwd,
            exclude=self._workspace_exclusions(output_path),
            timeout=self.config.workspace_fingerprint_timeout_seconds,
        )

    def save(self, path: Path | None, *extra_dicts) -> dict:
        data = super().save(path, *extra_dicts)
        if path:
            self.events.emit(
                "snapshot_written",
                path=str(path),
                n_messages=len(self.messages),
                workspace=self._fingerprint_workspace(path),
            )
        return data

    def serialize(self, *extra_dicts) -> dict:
        events = {
            "info": {
                "task": self._task,
                "tokens": _token_totals(self.messages),
                "events": {
                    "path": str(self.events.path) if self.events.path else None,
                    "run_id": self.events.run_id,
                    "schema_version": SCHEMA_VERSION,
                },
            }
        }
        return super().serialize(events, *extra_dicts)
