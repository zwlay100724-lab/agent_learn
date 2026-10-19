"""Agent that verifies a submission before accepting it."""

from pydantic import Field

from minisweagent.agents.resume import ResumableAgent, ResumeConfig


def derive_task_state(exit_status: str | None) -> str:
    """Map the loop's exit reason to an authoritative delivery state."""
    if exit_status == "Submitted":
        return "Unverified"
    if exit_status in ("LimitsExceeded", "TimeExceeded", "RepeatedFormatError"):
        return "Incomplete"
    return "Error"


def parse_git_status(output: str) -> list[str]:
    """Extract paths from `git status --porcelain`. Renames keep the new path."""
    paths = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        paths.append(line[3:].split(" -> ")[-1])
    return paths


class VerifiedAgentConfig(ResumeConfig):
    verify_command: str = ""
    """Command run after submission; returncode 0 means the delivery is accepted."""
    verify_max_attempts: int = Field(default=3, ge=1)
    """Maximum number of verification runs before accepting with VerificationFailed."""
    verify_feedback: bool = True
    """Feed a failed verification back to the agent and let it continue."""
    verify_feedback_max_chars: int = 4000
    """Inline at most this much verification output in the feedback message."""
    verify_timeout: int | None = None
    """Timeout for the verification command; defaults to the environment's timeout."""
    delivery_enabled: bool = True
    """Record the changed-file list when a submission is accepted."""
    delivery_command: str = "git status --porcelain"
    """Command used to collect delivery artifacts. A failure only records None."""


class VerifiedAgent(ResumableAgent):
    def __init__(self, *args, config_class: type = VerifiedAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self._verification_attempts = 0
        self._last_verification: dict | None = None
        self._delivery: dict | None = None

    def add_messages(self, *messages: dict) -> list[dict]:
        for message in messages:
            if message.get("role") == "exit":
                if message.get("extra") is None:
                    message["extra"] = {}
                message["extra"].setdefault("task_state", derive_task_state(message["extra"].get("exit_status")))
        return super().add_messages(*messages)

    def serialize(self, *extra_dicts) -> dict:
        last_extra = self.messages[-1].get("extra", {}) if self.messages else {}
        info = {
            "info": {
                "task_state": last_extra.get("task_state"),
                "verification": self._verification_info(),
                "delivery": self._delivery,
            }
        }
        return super().serialize(info, *extra_dicts)

    def _verification_info(self) -> dict | None:
        if not self._last_verification:
            return None
        return {"attempts": self._verification_attempts, "last": self._last_verification}