# Model setup quickstart

!!! tip "Setup"

    * Run `mini-extra config setup` once to set the default model and API key.
    * This project uses one model path: [`litellm`](https://github.com/BerriAI/litellm) with tool calling.

## Setting API keys

* **Recommended**: `mini-extra config setup`.
* `mini-extra config set ANTHROPIC_API_KEY <your-api-key>` writes the key to the
  [global config file](../advanced/global_configuration.md).
* Or export it in your shell: `export ANTHROPIC_API_KEY=<your-api-key>`.

LiteLLM supports many provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, and others). Use the key name
documented by your provider.

## Selecting a model

* `mini-extra config setup` sets the default model.
* Every CLI accepts `-m` / `--model`.
* `mini-extra config set TRACE_SWE_MODEL_NAME <model-name>` sets the default.
* A config file can set `model.model_name`.
* Include the provider in the name, e.g. `anthropic/claude-sonnet-4-5-20250929`,
  `openai/gpt-5.4`, or `deepseek/deepseek-flash`.

For local models, see the [local model guide](local_models.md).

??? note "List of all supported models"

    This list is generated from LiteLLM and may lag behind the latest release:

    ```
    --8<-- "docs/data/all_models.txt"
    ```

## Extra model settings

Set model-specific options under `model.model_kwargs` in the
[agent config file](../advanced/yaml_configuration.md):

=== "Temperature"

    ```yaml
    model:
      model_name: "anthropic/claude-sonnet-4-5-20250929"
      model_kwargs:
        temperature: 0.0
    ```

=== "Reasoning effort"

    ```yaml
    model:
      model_name: "openai/gpt-5.4"
      model_kwargs:
        reasoning_effort: "low"
    ```

=== "Custom OpenAI-compatible endpoint"

    ```yaml
    model:
      model_name: "openai/my-model"
      model_kwargs:
        api_base: "http://localhost:8000/v1"
        api_key: "EMPTY"
    ```

## Observation and format-error templates

`trace.yaml` defines `model.observation_template` and `model.format_error_template`.
Keep them aligned with tool calling: the model must call the `bash` tool with a
`command` string.

{% include-markdown "../_footer.md" %}
