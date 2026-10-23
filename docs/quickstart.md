# Quick start

## Install

This project is installed from source:

```bash
git clone https://github.com/zwlay100724-lab/trace-swe.git
cd trace-swe
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pre-commit install
pytest -q
```

## Configure a model

```bash
trace-extra config setup
```

Or set the model and key directly:

```bash
trace-extra config set TRACE_SWE_MODEL_NAME deepseek/deepseek-flash
trace-extra config set DEEPSEEK_API_KEY <your-key>
```

See [model setup](models/quickstart.md) for LiteLLM provider details.

## Run

```bash
trace -m deepseek/deepseek-flash \
  -t "Fix the failing test in this repository" -y
```

The CLI commands are:

```text
trace        # run the agent
trace-extra   # config, inspect, events, resume, eval, swebench
```

## Inspect and evaluate

```bash
trace --agent-class event_log -o run.traj.json -t "..." -y
trace-extra events run.traj.json

trace-extra eval runs/ --baseline baseline/
```

## Example prompts

- Implement a Sudoku solver in Python in the `sudoku` folder, with tests.
- Run pytest, find failing tests, and fix them.
- Add docstrings and type hints to a codebase.

## Next steps

- [Architecture](architecture.md)
- [Context experiment](experiments.md)
- [CLI usage](usage/mini.md)
