# trace-swe

A tracing-first coding agent with an append-only event log, offline evaluation,
context management, safe resume, and verified delivery.

The agent solves a task by issuing shell commands. Every run can be inspected,
measured, compared against a baseline, interrupted and resumed, and verified
before the result is accepted.

## Start here

<div class="grid cards" markdown>

- **[Quick start](quickstart.md)** — install, configure a model, and run the agent.
- **[Architecture](architecture.md)** — event log, agent layers, resume, and delivery.
- **[Context experiment](experiments.md)** — method, results, and interpretation.
- **[CLI usage](usage/mini.md)** — `mini` options and interactive modes.

</div>

## Features

- **Event log** — JSONL records for timings, retries, tokens, action outcomes,
  output hashes, and run summaries.
- **Offline evaluation** — strict success classification, failure buckets, token
  metrics, and baseline comparison.
- **Context management** — age out old observations after storing their full
  text on disk.
- **Resume safety** — workspace fingerprints at snapshots; conservative refusal
  for unconfirmed steps.
- **Verified delivery** — verify a submission, then write `verification.json`
  and `patch.diff`.
- **Reproducible experiments** — one command runs repeated variants and writes a
  report with git/model/task metadata.

## Scope

This build keeps one path per capability:

- Environments: `local`, `docker`
- Models: LiteLLM with native tool calling
- Benchmark: SWE-bench batch mode

## Quick start

```bash
pip install -e .
mini-extra config setup

mini -m deepseek/deepseek-flash \
  -t "Fix the failing test in this repository" -y
```

## License

MIT. See [LICENSE.md](LICENSE.md) for the copyright notice.

{% include-markdown "_footer.md" %}
