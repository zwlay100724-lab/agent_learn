"""Agent that verifies a submission before accepting it."""

from pydantic import Field

from minisweagent.agents.resume import ResumableAgent, ResumeConfig
from minisweagent.exceptions import Submitted


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

    def run(self, task: str = "", **kwargs) -> dict:
        self._verification_attempts = 0
        self._last_verification = None
        self._delivery = None
        return super().run(task, **kwargs)

    def execute_actions(self, message: dict) -> list[dict]:
        try:
            return super().execute_actions(message)
        except Submitted as e:
            feedback = self._verify_submission(e)
            if feedback is None:
                raise
            return self.add_messages(feedback)

    def _verify_submission(self, e: Submitted) -> dict | None:
        """Return None when the submission is accepted, or a feedback message to continue."""
        if not self.config.verify_command:
            self._accept(e, "Unverified")
            return None
        self._verification_attempts += 1
        result = self.env.check(
            self.config.verify_command,
            timeout=self.config.verify_timeout,
            attempt=self._verification_attempts,
        )
        self._last_verification = {k: result.get(k) for k in ("passed", "returncode", "output_ref", "exception_info")}
        if result["passed"] is None:
            self._accept(e, "Unverified")
            return None
        if result["passed"]:
            self._accept(e, "Succeeded")
            return None
        if not self.config.verify_feedback or self._verification_attempts >= self.config.verify_max_attempts:
            self._accept(e, "VerificationFailed")
            return None
        return self._feedback_message(result)

    def _accept(self, e: Submitted, task_state: str) -> None:
        message = e.messages[0]
        message.setdefault("extra", {})["task_state"] = task_state
        message["extra"]["verification"] = {
            "command": self.config.verify_command,
            "attempts": self._verification_attempts,
            "passed": (self._last_verification or {}).get("passed"),
            "returncode": (self._last_verification or {}).get("returncode"),
            "output_ref": (self._last_verification or {}).get("output_ref"),
        }
        self._collect_delivery()
        message["extra"]["delivery"] = self._delivery

    def _collect_delivery(self) -> None:
        if not self.config.delivery_enabled:
            self._delivery = None
            return
        result = self.env.check(self.config.delivery_command, event="delivery", offload=False)
        output = result.get("output", "") or ""
        self._delivery = {
            "command": self.config.delivery_command,
            "returncode": result.get("returncode"),
            "changed_files": parse_git_status(output) if result.get("returncode") == 0 else None,
        }

    def _feedback_message(self, result: dict) -> dict:
        output = result.get("output_preview", "")
        if result.get("output_ref"):
            output += f"\n(full output: {result['output_ref']})"
        return self.model.format_message(
            role="user",
            content=(
                f"Verification failed (attempt {self._verification_attempts}/"
                f"{self.config.verify_max_attempts}).\n"
                f"Command: {self.config.verify_command}\n"
                f"Return code: {result.get('returncode')}\n"
                f"Output:\n{output[: self.config.verify_feedback_max_chars]}\n"
                "Fix the issue and submit again with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
            ),
        )

    def _run_end_extra(self, last_extra: dict) -> dict:
        return {"task_state": last_extra.get("task_state")}

    def _run_end_success(self, last_extra: dict, exit_status: str | None) -> bool:
        return last_extra.get("task_state") == "Succeeded"

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