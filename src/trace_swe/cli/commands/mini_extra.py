#!/usr/bin/env python3

"""This is the central entry point to the mini-extra script. Use subcommands
to invoke other command line utilities like running on benchmarks, editing config,
inspecting trajectories, etc.
"""

import sys
from importlib import import_module

from rich.console import Console

subcommands = [
    ("trace_swe.cli.commands.config", ["config"], "Manage the global config file"),
    ("trace_swe.cli.commands.inspector", ["inspect", "i", "inspector"], "Run inspector (browse trajectories)"),
    ("trace_swe.cli.commands.events", ["events"], "Summarize an event log (timings, retries, cost, failures)"),
    ("trace_swe.cli.commands.resume", ["resume"], "Resume an interrupted run from its trajectory"),
    (
        "trace_swe.cli.commands.eval",
        ["eval"],
        "Evaluate saved trajectories offline (aggregate metrics, compare baseline)",
    ),
    ("trace_swe.benchmarks.swebench", ["swebench"], "Evaluate on SWE-bench (batch mode)"),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for all extra commands[/yellow] from trace-swe.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if len(args) == 0 or len(args) == 1 and args[0] in ["-h", "--help"]:
        return Console().print(get_docstring())

    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            return import_module(module_path).app(args[1:], prog_name=f"mini-extra {aliases[0]}")

    return Console().print(get_docstring())


if __name__ == "__main__":
    main()
