# Global configuration

!!! abstract "Configuring mini"

    * This guide shows how to configure the `mini` agent's global settings (API keys, default model, etc.).
      Basically anything that is set as environment variables or similar.
    * You should already be familiar with the [quickstart guide](../quickstart.md).
    * For more agent specific settings, see the [yaml configuration file guide](yaml_configuration.md).

!!! tip "Setting up models"

    Setting up models is also covered in the [quickstart guide](../quickstart.md).

## Setting global configuration

All global configuration can be either set as environment variables, or in the `.env` file (the exact location is printed when you run `mini`).
Environment variables take precedence over variables set in the `.env` file.

We provide several helper functions to update the global configuration.

For example, to set the default model and API keys, you can run:

```bash
mini-extra config setup
```

or to update specific settings:

```
mini-extra config set KEY VALUE
# e.g.,
mini-extra config set TRACE_SWE_MODEL_NAME "anthropic/claude-sonnet-4-5-20250929"
mini-extra config set ANTHROPIC_API_KEY "sk-..."
```

or to unset a key:

```bash
mini-extra config unset KEY
# e.g.,
mini-extra config unset ANTHROPIC_API_KEY
```

You can also edit the `.env` file directly and we provide a helper function for that:

```bash
mini-extra config edit
```

To set environment variables (recommended for temporary experimentation or API keys):

```bash
export KEY="value"
# windows:
setx KEY "value"
```

## Models, keys, costs

!!! tip "See also"

    Read the [quickstart guide](../quickstart.md) first—it already covers most of this.

```bash
# Default model name
# (default: not set)
TRACE_SWE_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
```

To ignore errors from cost tracking checks (for example for free models), set:

```bash
# CAREFUL: This can lead to unmanaged spending!
TRACE_SWE_COST_TRACKING="ignore_errors"
```

To register extra models to litellm (see [local models](../models/local_models.md) for more details), you can either specify the path in the agent file, or set

```bash
LITELLM_MODEL_REGISTRY_PATH="/path/to/your/model/registry.json"
```

Global cost limits:

```bash
# Global limit on number of model calls (0 = no limit)
# (default: 0)
TRACE_SWE_GLOBAL_CALL_LIMIT="100"

# Global cost limit in dollars (0 = no limit)
# (default: 0)
TRACE_SWE_GLOBAL_COST_LIMIT="10.00"

# Number of retry attempts for model API calls
# (default: 10)
TRACE_SWE_MODEL_RETRY_STOP_AFTER_ATTEMPT="10"
```

## Default config files

```bash
# Set a custom directory for agent config files in addition to the builtin ones
# This allows to specify them by names
TRACE_SWE_CONFIG_DIR="/path/to/your/own/config/dir"

# Config path for mini run script
# (default: package_dir / "config" / "trace.yaml")
TRACE_SWE_MINI_CONFIG_PATH="/path/to/your/own/config"

# Custom style path for trajectory inspector
# (default: package_dir / "config" / "inspector.tcss")
TRACE_SWE_INSPECTOR_STYLE_PATH="/path/to/your/inspector/style.tcss"
```

### Settings for environments

```bash
# Path/name to the docker executable
# (default: "docker")
TRACE_SWE_DOCKER_EXECUTABLE="docker"
```

## Default run files

```bash
# Default run script entry point for the main CLI
# (default: "trace_swe.cli.trace")
TRACE_SWE_DEFAULT_RUN="trace_swe.cli.trace"
```

{% include-markdown "_footer.md" %}
