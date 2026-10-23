# `trace`

!!! abstract "Overview"

    * `trace` is the primary REPL-style interactive command line interface for using trace-swe in the local environment. `mini` remains an alias.
    * The agent is intended for local work or sandboxed execution; it is not a batch runner.


## Command line options

Useful switches:

- `-h`/`--help`: Show help
- `-t`/`--task`: Specify a task to run (else you will be prompted)
- `-c`/`--config`: Specify a config file to use, else we will use [`trace.yaml`](https://github.com/zwlay100724-lab/trace-swe/blob/main/src/trace_swe/config/trace.yaml) or the config `TRACE_SWE_MINI_CONFIG_PATH` environment variable (see [global configuration](../advanced/global_configuration.md)).
  It's enough to specify the name of the config file, e.g., `-c trace.yaml` (see [global configuration](../advanced/global_configuration.md) for how it is resolved).
- `-m`/`--model`: Specify a model to use, else we will use the model `TRACE_SWE_MODEL_NAME` environment variable (see [global configuration](../advanced/global_configuration.md))
- `-y`/`--yolo`: Start in `yolo` mode (see below)

## Modes of operation

`mini` provides three different modes of operation

- `confirm` (`/c`): The LM proposes an action and the user is prompted to confirm (press Enter) or reject (enter a rejection message)
- `yolo` (`/y`): The action from the LM is executed immediately without confirmation
- `human` (`/u`): The user takes over to type and execute commands

You can switch between the modes with the `/c`, `/y`, and `/u` commands that you can enter any time the agent is waiting for input.
You can also press `Ctrl+C` to interrupt the agent at any time, allowing you to switch between modes.

`mini` starts in `confirm` mode by default. To start in `yolo` mode, you can add `-y`/`--yolo` to the command line.

## Miscellaneous tips

- `mini` saves the full history of your last run to your global config directory.
  The path to the directory is printed when you start `mini`.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/zwlay100724-lab/trace-swe/blob/main/src/trace_swe/config/trace.yaml)

    ```yaml
    --8<-- "src/trace_swe/config/trace.yaml"
    ```

??? note "Run script"

    - [Read on GitHub](https://github.com/zwlay100724-lab/trace-swe/blob/main/src/trace_swe/cli/trace.py)
    - [API reference](../reference/run/trace.md)

    ```python
    --8<-- "src/trace_swe/cli/trace.py"
    ```

??? note "Agent class"

    - [Read on GitHub](https://github.com/zwlay100724-lab/trace-swe/blob/main/src/trace_swe/agent/interactive.py)
    - [API reference](../reference/agents/interactive.md)

    ```python
    --8<-- "src/trace_swe/agent/interactive.py"
    ```

{% include-markdown "../_footer.md" %}