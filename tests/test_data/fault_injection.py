"""Run a two-step ResumableAgent and hard-exit at a selected point (test helper)."""

import os
from pathlib import Path

from trace_swe.agent.default import DefaultAgent
from trace_swe.agent.resume import ResumableAgent
from trace_swe.execution.local import LocalEnvironment
from trace_swe.llm.test_models import DeterministicModel, make_output

CRASH = 17


class CrashBeforeAction(ResumableAgent):
    def execute_actions(self, message):
        if self._step_index >= 2:
            os._exit(CRASH)
        return super().execute_actions(message)


class CrashAfterAction(ResumableAgent):
    def execute_actions(self, message):
        if self._step_index >= 2:
            super().execute_actions(message)
            os._exit(CRASH)
        return super().execute_actions(message)


class CrashBeforeSnapshot(ResumableAgent):
    def save(self, path, *extra_dicts):
        if self._step_index >= 2:
            DefaultAgent.save(self, path, *extra_dicts)
            os._exit(CRASH)
        return super().save(path, *extra_dicts)


AGENTS = {
    "before_action": CrashBeforeAction,
    "after_action": CrashAfterAction,
    "before_snapshot": CrashBeforeSnapshot,
}


def main() -> None:
    agent = AGENTS[os.environ["FAULT_POINT"]](
        model=DeterministicModel(
            outputs=[
                make_output("", [{"command": "echo one >> log.txt"}]),
                make_output("", [{"command": "echo two >> log.txt"}]),
                make_output("", [{"command": "echo done"}]),
            ]
        ),
        env=LocalEnvironment(cwd=os.environ["FAULT_CWD"]),
        system_template="system",
        instance_template="{{task}}",
        mode="yolo",
        confirm_exit=False,
        output_path=Path(os.environ["FAULT_OUT"]),
    )
    agent.run("task")


if __name__ == "__main__":
    main()
