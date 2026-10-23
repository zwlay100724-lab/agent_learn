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
.venv/bin/python experiments/context_tasks/run_experiment.py --runs 5
```

The runner writes `experiments/context_tasks/results/meta.json` (git commit, model, temperature,
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

Run on 2026-10-22, commit `72b0270`, task-set hash
`b5cb255b1fe29eeaaf77f9913ccd38ee812394179e3e3d938a7979102d17c408`, 5 runs per cell.

### Overall

| Variant | Success | Prompt tokens/call | Total prompt tokens (mean) | Steps (mean) | Cost (mean) | Wall (mean) | Elisions |
|---|---|---|---|---|---|---|---|
| `event_log` | 10/10 | 3350 | 20,103 | 6.00 | $0.00269 | 13.54s | 0 |
| `context` | 10/10 | 2875 | 12,651 | 4.40 | $0.00279 | 13.16s | 15 |

Prompt tokens per call dropped **14.2%**; total prompt tokens dropped **37.1%**, but part of that is
the lower step count (**-26.7%**). Cost did not improve (**+3.9%**, within the observed spread), so
the result is an efficiency result, not a cost result.

### Per task

| Task | Variant | Success | Prompt tokens/call | Total prompt tokens (mean) | Steps | Cost | Wall |
|---|---|---|---|---|---|---|---|
| `sum_range` | `event_log` | 5/5 | 1723 | 8,615 | 5.00 | $0.00118 | 8.4s |
| `sum_range` | `context` | 5/5 | 1609 | 6,436 | 4.00 | $0.00118 | 11.4s |
| `log_errors` | `event_log` | 5/5 | 4513 | 31,590 | 7.00 | $0.00419 | 18.7s |
| `log_errors` | `context` | 5/5 | 3930 | 18,866 | 4.80 | $0.00441 | 14.9s |

The long-log task is where elision matters: **-12.9% prompt tokens per call**, **-31.4% steps**,
and **-20.4% wall time**. On the small task the effect is smaller (**-6.6% prompt tokens per call**),
and wall time is noisy (11.4s vs 8.4s).

### Distribution

| Task | Variant | Prompt tokens/call, median (range) | Steps, median (range) |
|---|---|---|---|
| `sum_range` | `event_log` | 1645 (1590–2035) | 5 (4–6) |
| `sum_range` | `context` | 1579 (1437–1798) | 4 (3–5) |
| `log_errors` | `event_log` | 4181 (2515–7121) | 7 (5–9) |
| `log_errors` | `context` | 3891 (3618–4265) | 5 (4–5) |

The context variant is also more consistent on the long task: its prompt-token range is much
narrower than the baseline's, and every run takes 4–5 steps instead of 5–9.

## Interpretation

- **Success is preserved**: 10/10 in both variants, and no protected file was modified.
- **Elision is reaching the model**: 15 elisions removed 67,948 characters in the context runs.
- **The benefit scales with observation size**: large effect on `log_errors`, small effect on
  `sum_range`.
- **Cost is not a win**: fewer steps and lower prompt tokens did not translate into lower cost.
  Do not claim cost savings from this experiment.
- **Wall time is noisy**: the long-task improvement is plausible; the small-task regression is not
  conclusive.

## Threats to validity

- Two tasks are not representative of SWE-bench-scale work, and n=5 per cell is small.
- Temperature 0 does not make model output fully deterministic.
- The elision threshold (200 chars, keep 2) is deliberately aggressive to expose effects; the code
  default is 2000 chars and 8 observations.
- The scorer checks behavior with a specific test command; a different verifier could change the
  success labels.
- Cost is dominated by cache and provider pricing effects, not only by prompt length.

## Decision

- **Keep observation elision enabled** for long-running tasks; it preserves success and reduces
  prompt tokens per call and step count.
- **Do not implement C4 (LLM summarization) yet.** The experiment does not show that observation
  elision is insufficient. Revisit C4 only after a larger task set shows a failure mode.
- **Do not claim cost savings.** The observed cost delta is slightly positive.
- **C5 (task-fact preservation) remains conditional.** No task facts were lost in this run; if a
  larger experiment shows lost goals or constraints, implement C5 before C4.
