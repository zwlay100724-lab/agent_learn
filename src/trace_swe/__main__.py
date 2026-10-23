#!/usr/bin/env python3
"""This is what happens when you do `python -m trace_swe` or `pipx run trace-swe`."""

from trace_swe.cli.trace import app

if __name__ == "__main__":
    app()
