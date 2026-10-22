# Context-management experiment

This page documents the experiment behind `ContextManagedAgent`: does aging out old
observations reduce prompt tokens without hurting task success?

## Question

When an agent runs for many steps, old command outputs dominate the prompt. `ContextManagedAgent`
keeps the most recent observations in full and replaces older, long observations with a path to
their stored full text. The experiment measures whether that trade is worthwhile.

## Setup

| Item | Value |
|---|---|
| Model | `deepseek/deepseek-flash` |
| Temperature | `0` |
| Tasks | `sum_range` (small fix), `log_errors` (large log, multi-file) |
| Variants | `event_log` (baseline), `context` (elision enabled) |
| Runs per task and variant | `5` |
| Step limit | `12` |
| Cost limit per run | `$0.05` |
| Context config | keep `2` observations, elide from `200` chars |

The `context` variant is the baseline plus:

```text
agent.context_max_observations=2
agent.context_elide_min_chars=200
```

The scorer runs `pytest` in the task repository and rejects submissions that modify
`test_buggy.py` or `app.log`, so an agent cannot pass by changing the tests.

## Reproduce

```bash
.venv/bin/python examples/eval_tasks/run_experiment.py --runs 5
```

The runner writes `examples/eval_tasks/results/meta.json` (git commit, model, temperature,
task-set hash, command), one directory per run, and `report.json` with the baseline comparison.
It also prints a human-readable table from `mini-extra eval`.

## Metrics

- **Strict success rate**: only trajectories with `task_state == "Succeeded"` count.
- **Prompt tokens per call**: total prompt tokens divided by calls that reported usage. This is the
  primary compression metric; total tokens per run is confounded by step count.
- **Total prompt tokens**, completion tokens, cost.
- **Steps**, wall time, and retries.
- **Elisions / elided chars** reported by the context agent.

## Results

> Run the command above, then replace this section with the output of `report.json`.

| Variant | Success | Prompt tokens/call | Total prompt tokens | Cost | Steps |
|---|---|---|---|---|---|
| `event_log` | TBD | TBD | TBD | TBD | TBD |
| `context` | TBD | TBD | TBD | TBD | TBD |

## Interpretation rules

- If success drops, the compression is too aggressive regardless of token savings.
- If prompt tokens per call do not drop, elision is not reaching the model (check `info.context`
  and the `context_elide` events).
- n=5 is small. Report the distribution, not only the mean, and do not treat a difference smaller
  than the observed spread as a win.

## Threats to validity

- Two tasks are not representative of SWE-bench-scale work.
- Temperature 0 does not make model output fully deterministic.
- The elision threshold (200 chars, keep 2) is deliberately aggressive to expose effects.
- The scorer checks behavior with a specific test command; a different verifier could change the
  success labels.

## Decision

- If `context` keeps success and lowers prompt tokens per call, keep elision on by default.
- If it loses task facts or increases failure rate, implement task-fact preservation (C5) before
  considering LLM summarization (C4).
- If neither variant shows a difference, stop at measurement and do not add summarization.
