# Architecture

trace-swe keeps the minimal `DefaultAgent` loop and adds observability, context management,
resume, verification, and evaluation as layers on top. The shared substrate is an append-only
event log.

## Layers

```text
DefaultAgent
  └── InteractiveAgent
        └── EventLogAgent              # event log + large-output offload
              ├── ResumableAgent       # resume from a saved trajectory
              │     └── VerifiedAgent  # verify a submission before accepting it
              └── ContextManagedAgent  # age out old observations
```

Run scripts select a class with `--agent-class` (or `agent.agent_class` in YAML), so the default
`mini` behavior is unchanged unless a feature is requested.

## Event log

`EventLogAgent` wraps the model and environment in transparent proxies. The agent loop, messages,
and trajectory format are not modified; the log is a side channel.

`EventLogWriter` writes one JSON object per line to `<output_path>.events.jsonl` (or
`event_log_path`). Every line is flushed, so a crash leaves a valid prefix. `read_events()` tolerates
a truncated final line but rejects corruption in the middle.

Every event carries:

| Field | Meaning |
|---|---|
| `v` | Schema version |
| `seq` | Monotonic sequence within the file |
| `ts`, `mono` | Wall-clock and monotonic timestamps |
| `run_id` | Groups a run, including resumed segments |
| `step` | Agent step index |
| `event` | Event type |

Event types:

| Event | Emitted when | Notable fields |
|---|---|---|
| `run_start` | `run()` begins | agent/model/env types, paths, version |
| `run_resumed` | A trajectory is resumed | trajectory, clean boundary, in-flight events, probe |
| `step_start` / `step_end` | Around each `step()` | duration, outcome, exit status |
| `llm_call` | Each model call | duration, attempts, cost, tokens, digest, action count |
| `llm_retry` | Tenacity retries a call | attempt number, error |
| `action` | Each environment action | duration, returncode, submitted, output size/hash/preview/ref |
| `verification` | A non-agent check command | command, passed, returncode, output |
| `delivery` | Delivery artifacts are collected | command, returncode, output |
| `workspace_probe` | A resume check command | command, returncode, output hash |
| `context_elide` | Old observations are aged out | count, chars, per-message paths |
| `snapshot_written` | The trajectory is saved | path, message count |
| `run_end` | The run exits | exit status, task state, cost, tokens, error |

Token usage is normalized by `models/utils/usage.py`. It handles completion APIs
(`prompt_tokens` / `completion_tokens`) and response APIs (`input_tokens` / `output_tokens`), plus
Anthropic cache fields. Missing usage is `None`, never `0`. Format errors are billed, so the
`FormatError` path records usage too. Each message keeps its normalized usage in `extra.usage`;
`info.tokens` in the trajectory is recomputed from those messages, so resuming a run does not lose
the totals.

## Large outputs

`_EventLogEnvironment` stores an action output longer than `output_offload_threshold_chars`
(default 10,000) under `<event log>.outputs/<run_id>/<seq>-<label>.txt`, capped at
`output_offload_max_bytes` (default 5 MB). The observation template can point the model at
`output_path`, so it can read the full text with `grep`, `sed`, or `head` instead of carrying it in
the context. Offload is limited to `LocalEnvironment`; other environments keep their output inline.

## Context management

`ContextManagedAgent` tags observation messages as they are created. Before each model call it
keeps the most recent `context_max_observations` observations and replaces older observations
longer than `context_elide_min_chars` with:

```text
[observation elided to save context; full output: <path>]
```

The full text is stored under `<event log>.context/<run_id>/`. Elision is idempotent: a message
marked `extra.elided` is skipped. System, task, assistant, and exit messages are never elided.
Each compaction emits a `context_elide` event, and the trajectory records
`info.context.elided_observations` and `elided_chars`.

The experiment uses an aggressive configuration (`max=2`, `min_chars=200`) to expose the effect;
the code default is `max=8`, `min_chars=2000`.

## Resume

`ResumableAgent` loads messages, cost, calls, and task from the trajectory, then derives the run
boundary from the event log:

- It refuses to resume a run that already ended with an exit message or `run_end`.
- It refuses by default when the last event is not `snapshot_written` or `run_end`, because the
  in-flight step may already have changed the workspace. The error lists the commands recorded
  after the last snapshot.
- `resume_policy="as_is"` proceeds anyway, accepting at-least-once semantics.
- `resume_check_command` runs a workspace probe before continuing and records it as
  `workspace_probe`.

Each `snapshot_written` also records a workspace fingerprint for `LocalEnvironment`: the git HEAD,
a content tree hash built with a temporary index (so the real index is untouched), and a dirty
flag. Agent artifacts such as the trajectory, event log, offloaded outputs, and patch are excluded.
On resume, the current tree is compared with the snapshot's tree:

- A mismatch refuses by default with both tree hashes.
- `resume_policy="as_is"` proceeds and records `workspace_changed: true` in `run_resumed`.
- A non-git or remote workspace is `unavailable`; the resume proceeds on the event boundary alone.

The tests simulate three hard-crash windows (before an action, after an action but before the
result is saved, and after the result is saved but before the snapshot) and assert that resume
refuses and does not replay the recorded action.

The loop itself is the extracted `DefaultAgent._run_loop()`, so a resumed agent re-enters the same
control flow. A resumed run keeps the original `run_id` and continues the event `seq`.

## Verification and delivery

`VerifiedAgent` intercepts a `Submitted` exit and runs `verify_command` out of band. Return code 0
accepts the submission as `Succeeded`. A failure is fed back to the model (up to
`verify_max_attempts`) with the command and output; exhausting attempts accepts it as
`VerificationFailed`. If the verifier itself crashes, the result is `Unverified`.

The exit status is unchanged; the authoritative delivery state is `task_state`:

| State | Meaning |
|---|---|
| `Succeeded` | Verified successfully |
| `VerificationFailed` | Verification ran and failed |
| `Unverified` | Submitted without a verifier, or the verifier could not run |
| `Incomplete` | Limits or repeated format errors |
| `Error` | Other termination |

On acceptance, `delivery_command` (default `git status --porcelain`) records the changed files in
`info.delivery`. For `LocalEnvironment`, the agent also writes:

- `<trajectory>.verification.json`: command, return code, duration, output reference, pass/fail,
  attempts, and `task_state`. It is written from the same data as `info.verification`.
- `<trajectory>.patch.diff`: a binary-safe patch against `HEAD`, including untracked files. The
  patch is built with a temporary git index (`GIT_INDEX_FILE`) so the workspace's real index is
  not modified. It is skipped when the workspace is not a git repository, when the environment is
  not local, or when it exceeds `patch_max_bytes`.

## Evaluation

`mini-extra eval` reads trajectories offline and reports:

- Strict success rate: only `task_state == "Succeeded"` counts.
- Outcome buckets and failure reasons.
- Cost, wall/LLM/environment time, steps, calls, and retries.
- Prompt/completion/total tokens, including mean prompt tokens per call.
- Elisions and elided characters.
- Coverage warnings for missing event logs or token usage.

`--baseline` compares two directories overall and by task group. Token deltas use absolute values
and percentages; success uses percentage points. Missing metrics stay `None` and render as `n/a`,
so they are not confused with zero.

## Reproducible experiments

`experiments/context_tasks/run_experiment.py` runs each task and variant in its own copied git repository,
scores it with an external `pytest`, and writes `meta.json` plus `report.json`. Metadata includes
the git commit and dirty flag, model, temperature, task-set hash, command, and run count. See
[Context-management experiment](experiments.md).

## Known limitations

- The agent hierarchy is single-inheritance: `ContextManagedAgent` and `VerifiedAgent` cannot be
  combined without a mixin or wrapper refactor.
- Benchmark runners construct `ProgressTrackingAgent(DefaultAgent)` directly, so they do not emit
  event logs yet.
- Workspace fingerprints and patch export are limited to local git worktrees; remote environments
  and non-git workspaces fall back to the event-boundary check alone.
- Patch export is limited to `LocalEnvironment`; remote environments need their own export path.
- If `output_path` is inside the workspace, the trajectory and event log appear in the changed-file
  list and patch. Keep outputs outside the workspace.
- The event schema is additive at version 1; new fields are optional so older logs remain readable.
