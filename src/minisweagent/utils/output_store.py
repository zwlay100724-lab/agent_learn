"""Store large command outputs on disk so the model can retrieve them on demand."""

import logging
from pathlib import Path

logger = logging.getLogger("minisweagent")


class OutputStore:
    def __init__(self, root: Path | None, max_bytes: int):
        self.root = root
        self.max_bytes = max_bytes

    def save(self, text: str, name: str) -> Path | None:
        """Write `text` under `root/name`, truncating at max_bytes. Returns the path or None."""
        if self.root is None:
            return None
        path = self.root / name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = text.encode("utf-8", errors="replace")
            if len(data) > self.max_bytes:
                text = data[: self.max_bytes].decode("utf-8", errors="replace")
                text += f"\n\n[mini-swe-agent: output truncated at {self.max_bytes} bytes]\n"
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not store output at '{path}': {e}")
            return None
        return path