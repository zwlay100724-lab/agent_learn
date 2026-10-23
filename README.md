# trace-swe

A tracing-first coding agent with an append-only event log, offline evaluation,
context management, safe resume, and verified delivery.

The agent solves a task by issuing shell commands. Every run can be inspected,
measured, compared against a baseline, interrupted and resumed, and verified
before the result is accepted.

## Features

### Append-only event log

Each run writes a JSONL event log next to its trajectory:

- per-step timing
- LLM call duration, attempts, and retries
- action return codes and output hashes
- prompt/completion/total tokens
- run summary and failure reasons

The log is a side channel: it does not change the agent loop or the trajectory.

```bash
trace --agent-class event_log \
  -m deepseek/deepseek-flash \
  -o run.traj.json \
  -t "Fix the failing test" -y

trace-extra events run.traj.json
```

### Offline evaluation

Evaluate saved trajectories with strict success classification and baseline
comparison:

```bash
trace-extra eval runs/ --baseline baseline/
```

Reports include:

- success rate (only `task_state == "Succeeded"` counts)
- outcome and failure buckets
- cost, wall time, LLM time, environment time
- prompt/completion/total tokens
- mean prompt tokens per call
- context elisions and elided characters
- retries and coverage warnings

### Context management

Old observations can be aged out after their full text is stored on disk:

```bash
trace --agent-class context \
  -m deepseek/deepseek-flash \
  -o context.traj.json \
  -t "Fix the failing test" -y
```

The agent keeps the most recent observations in full and replaces older ones
with a path to the stored output. Each compaction is recorded as a
`context_elide` event.

### Resume safety

A snapshot records a workspace fingerprint. Resume refuses to continue when:

- the last event is not a snapshot, or
- the workspace changed since the last snapshot.

```bash
trace-extra resume run.traj.json --policy refuse
```

Use `--policy as_is` to proceed anyway; the run records
`workspace_changed: true`.

### Verified delivery

A submission can be verified before it is accepted:

```bash
trace --agent-class verified \
  -m deepseek/deepseek-flash \
  -c mini.yaml \
  -c "agent.verify_command=pytest -q" \
  -t "Fix the failing test" -y
```

On acceptance, the agent writes:

```text
<trajectory>.verification.json
<trajectory>.patch.diff
```

The patch includes untracked files and is built with a temporary git index, so
the workspace's real index is not modified.

## Quick start

```bash
pip install -e .
trace-extra config setup

trace -m deepseek/deepseek-flash \
  -t "Fix the failing test in this repository" -y
```

The CLI commands are `trace` and `trace-extra`; `mini` and `mini-extra` remain aliases.

## Reproducible experiment

Run the context-management experiment:

```bash
python experiments/context_tasks/run_experiment.py --runs 5
```

Result from 2 tasks × 5 runs × 2 variants with `deepseek/deepseek-flash`:

| Metric | baseline | context |
|---|---:|---:|
| Success | 10/10 | 10/10 |
| Prompt tokens/call | 3,350 | 2,875 |
| Total prompt tokens | 201,026 | 126,510 |
| Mean steps | 6.00 | 4.40 |
| Mean cost | $0.00269 | $0.00279 |

Observation elision preserved success and reduced prompt tokens per call by
14.2%. It did not reduce cost. Details are in
[docs/experiments.md](docs/experiments.md).

## Scope

This build keeps one path per capability:

- **Environments**: `local`, `docker`
- **Models**: LiteLLM with native tool calling
- **Benchmark**: SWE-bench batch mode

## Project layout

```text
src/trace_swe/agent/        agent loop, event log, context, resume, verification
src/trace_swe/execution/    local and docker execution
src/trace_swe/llm/          LiteLLM tool-calling model
src/trace_swe/cli/          trace, trace-extra
src/trace_swe/benchmarks/   SWE-bench batch runner
src/trace_swe/infra/        event log, output store, workspace fingerprint
experiments/context_tasks/  reproducible context experiment
docs/                       usage, architecture, experiment results
tests/                      test suite
```

## Testing

```bash
pytest -q
ruff check src tests
```

## Limitations

- Event log and workspace fingerprint are local-first.
- Resume is conservative by default; `as_is` is at-least-once.
- Patch export is limited to local git worktrees.
- Docs build currently depends on a fixed mkdocstrings/pymdown combination.

## License

MIT. See [LICENSE.md](LICENSE.md) for the copyright notice.
