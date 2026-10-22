"""Context management: age out old observations after storing their full text."""

from pathlib import Path

from minisweagent.agents.event_log import EventLogAgent, EventLogAgentConfig
from minisweagent.utils.output_store import OutputStore


class ContextConfig(EventLogAgentConfig):
    context_elide_enabled: bool = True
    """Replace old observations with a reference to their stored full text."""
    context_max_observations: int = 8
    """Keep this many of the most recent observations in full."""
    context_elide_min_chars: int = 2000
    """Only elide observations at least this long."""


class _TaggingModel:
    """Mark messages produced by format_observation_messages so they can be aged out later."""

    def __init__(self, model):
        self._model = model

    def format_observation_messages(self, *args, **kwargs):
        messages = self._model.format_observation_messages(*args, **kwargs)
        for message in messages:
            message.setdefault("extra", {})["observation"] = True
        return messages

    def __getattr__(self, name):
        return getattr(self._model, name)


def _message_text(message: dict) -> str | None:
    if message.get("type") == "function_call_output":
        return message.get("output") if isinstance(message.get("output"), str) else None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _set_message_text(message: dict, text: str) -> None:
    if message.get("type") == "function_call_output":
        message["output"] = text
    else:
        message["content"] = text


class ContextManagedAgent(EventLogAgent):
    def __init__(self, *args, config_class: type = ContextConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.model = _TaggingModel(self.model)
        self._context_elided_observations = 0
        self._context_elided_chars = 0

    def _context_store(self) -> OutputStore | None:
        if self.events.path is None:
            return None
        return OutputStore(Path(f"{self.events.path}.context"), self.config.output_offload_max_bytes)

    def run(self, task: str = "", **kwargs) -> dict:
        self._context_elided_observations = 0
        self._context_elided_chars = 0
        return super().run(task, **kwargs)

    def query(self) -> dict:
        self._compact_context()
        return super().query()

    def _compact_context(self) -> None:
        if not self.config.context_elide_enabled:
            return
        observations = [
            (index, message)
            for index, message in enumerate(self.messages)
            if message.get("extra", {}).get("observation") and not message.get("extra", {}).get("elided")
        ]
        keep = max(0, self.config.context_max_observations)
        candidates = observations[:-keep] if keep else observations
        store = self._context_store()
        elided = []
        for index, message in candidates:
            text = _message_text(message)
            if text is None or len(text) < self.config.context_elide_min_chars:
                continue
            path = message.get("extra", {}).get("output_path")
            if not (path and Path(path).is_file()):
                if store is None:
                    continue
                saved = store.save(text, f"{self.events.run_id}/obs-{index:06d}.txt")
                if saved is None:
                    continue
                path = str(saved)
            _set_message_text(message, f"[observation elided to save context; full output: {path}]")
            message.setdefault("extra", {})["elided"] = True
            self._context_elided_observations += 1
            self._context_elided_chars += len(text)
            elided.append({"index": index, "path": path, "chars": len(text)})
        if elided:
            self.events.emit(
                "context_elide",
                count=len(elided),
                chars_elided=sum(item["chars"] for item in elided),
                observations=elided,
            )

    def serialize(self, *extra_dicts) -> dict:
        info = {
            "info": {
                "context": {
                    "elided_observations": self._context_elided_observations,
                    "elided_chars": self._context_elided_chars,
                }
            }
        }
        return super().serialize(info, *extra_dicts)
