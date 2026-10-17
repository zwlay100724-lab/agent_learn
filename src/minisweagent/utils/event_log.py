"""Append-only JSONL event log for observing an agent run."""

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

logger = logging.getLogger("minisweagent")

SCHEMA_VERSION = 1


def digest(value: Any) -> str:
    """Stable hash of a JSON-serializable value, used to correlate events with messages."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class EventLogWriter:
    """Writes one JSON object per line. Each line is flushed, so a crash leaves a valid prefix."""

    def __init__(self, path: Path | None, run_id: str | None = None):
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex
        self.seq = 0
        self.step = 0
        self.retry_count = 0
        self._file = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def new_run(self, run_id: str | None = None, start_seq: int = 0) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.seq = start_seq
        self.step = 0
        self.retry_count = 0

    def emit(self, event: str, **payload) -> dict | None:
        if not self.enabled:
            return None
        data = {
            **payload,
            "v": SCHEMA_VERSION,
            "seq": self.seq,
            "ts": time.time(),
            "mono": time.monotonic(),
            "run_id": self.run_id,
            "step": self.step,
            "event": event,
        }
        try:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("a", encoding="utf-8")
            self._file.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except OSError as e:
            # Never mask the agent's real exception with a logging failure
            logger.warning(f"Disabling event log '{self.path}': {e}")
            self.path = None
            self._file = None
            return None
        self.seq += 1
        return data

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def read_events(path: Path) -> list[dict]:
    """Read an event log, tolerating a truncated final line but not corruption in the middle."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return events


_current_event_log: ContextVar[EventLogWriter | None] = ContextVar("minisweagent_event_log", default=None)


@contextmanager
def bind_event_log(writer: EventLogWriter):
    """Make the writer visible to nested code (e.g. the model retry loop)."""
    token = _current_event_log.set(writer)
    try:
        yield writer
    finally:
        _current_event_log.reset(token)


def current_event_log() -> EventLogWriter | None:
    return _current_event_log.get()
