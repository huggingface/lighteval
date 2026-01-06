# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from collections import defaultdict
from datetime import datetime
from typing import Literal

import requests
from huggingface_hub import HfApi
from inspect_ai import Epochs, Task, task
from inspect_ai import eval_set as inspect_ai_eval_set
from inspect_ai.dataset import hf_dataset
from inspect_ai.log import bundle_log_dir
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message
from pytablewriter import MarkdownTableWriter
from typer import Argument, Option
from typing_extensions import Annotated

from lighteval.models.abstract_model import InspectAIModelConfig
from lighteval.tasks.lighteval_task import LightevalTaskConfig


logger = logging.getLogger(__name__)


@task
def get_inspect_ai_task(
    lighteval_task_config: LightevalTaskConfig,
    epochs: int = 1,
    epochs_reducer: Literal["mean", "median", "mode", "max", "at_least_{n}", "pass_at_{k}"] | None = None,
) -> Task:
    name = lighteval_task_config.name
    sample_fields = lighteval_task_config.sample_fields

    if sample_fields is None:
        raise ValueError(
            f"Task {name} is not supported by inspect_ai yet. You can either define it or use a different backend, `lighteval --help`"
        )

    dataset_repo = lighteval_task_config.hf_repo
    dataset_subset = lighteval_task_config.hf_subset
    dataset_split = lighteval_task_config.evaluation_splits[0]

    dataset = hf_dataset(dataset_repo, name=dataset_subset, split=dataset_split, sample_fields=sample_fields)
    if lighteval_task_config.filter is not None:
        dataset = dataset.filter(lighteval_task_config.filter)
    solver = lighteval_task_config.solver or [
        generate(cache=True),
    ]
    scorers = lighteval_task_config.scorer or exact()
    # TODO: have per task epoch and epoch reducer

    if lighteval_task_config.num_fewshots > 0:
        name += f"_{lighteval_task_config.num_fewshots}_shots"
        # TODO: use fewshot split
        fewshots = hf_dataset(
            path=dataset_repo,
            name=dataset_subset,
            split=dataset_split,
            sample_fields=sample_fields,
            shuffle=True,
            seed=42,
            limit=lighteval_task_config.num_fewshots,
        )
        solver.insert(
            0,
            system_message("\n\n".join([lighteval_task_config.sample_to_fewshot(sample) for sample in fewshots])),
        )

    return Task(dataset=dataset, solver=solver, scorer=scorers, name=name, epochs=Epochs(epochs, epochs_reducer))


def push_to_hub(bundle_dir: str, repo_id: str, public: bool = False):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True, private=not public)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=bundle_dir)
    print(f"Details pushed to https://huggingface.co/spaces/{repo_id}")


def mean_metrics_by_prefix(results_per_model_per_task, sep=":"):
    out = {}
    for model, tasks in results_per_model_per_task.items():
        pref_metrics = defaultdict(lambda: defaultdict(list))
        # Collect both per-task metrics and values for prefix aggregation
        per_model_out = {}
        for task_name, metrics in tasks.items():
            if sep not in task_name:
                # No subtasks: keep metrics as-is for this task
                per_task_vals = {}
                for mname, metric in metrics.items():
                    per_task_vals[mname] = getattr(metric, "value", metric)
                per_model_out[task_name] = per_task_vals
                continue
            prefix = task_name.split(sep, 1)[0]
            # Keep non-averaged task metrics
            per_task_vals = {}
            for mname, metric in metrics.items():
                value = getattr(metric, "value", metric)
                per_task_vals[mname] = value
                pref_metrics[prefix][mname].append(value)
            per_model_out[task_name] = per_task_vals
        # Add the averaged metrics per prefix
        for p, md in pref_metrics.items():
            per_model_out[p] = {m: sum(v) / len(v) for m, v in md.items()}
        out[model] = per_model_out
    return out


def results_to_markdown_table(
    results_per_model_per_task,
    metric: str = "accuracy",
    stderr_metric: str = "stderr",
    max_total_columns: int | None = None,
    means_only_task_threshold: int = 10,
) -> str:
    cols = _collect_columns(results_per_model_per_task, means_only_task_threshold, max_total_columns)

    writer = MarkdownTableWriter()
    writer.headers = ["Model"] + cols

    rows = []
    for model in sorted(results_per_model_per_task.keys()):
        row = [model]
        data = results_per_model_per_task[model]
        for col in cols:
            row.append(_format_metric_cell(data, col, metric, stderr_metric))
        rows.append(row)

    writer.value_matrix = rows
    return writer.dumps()


def _collect_columns(
    results_per_model_per_task, means_only_task_threshold: int, max_total_columns: int | None
) -> list[str]:
    all_cols = set()
    for model_data in results_per_model_per_task.values():
        all_cols.update(model_data.keys())
    agg_cols = sorted([c for c in all_cols if ":" not in c])
    task_cols = sorted([c for c in all_cols if ":" in c])

    if len(task_cols) > means_only_task_threshold:
        logger.info(
            f"Only showing the meaned tasks (aggregates only) because there are more than {means_only_task_threshold} tasks"
        )
        return agg_cols

    cols = agg_cols + task_cols
    if max_total_columns is not None and len(cols) > max_total_columns:
        keep_left = max(1, max_total_columns // 2)
        keep_right = max_total_columns - keep_left
        left_cols = cols[:keep_left]
        right_cols = cols[-keep_right:] if keep_right > 0 else []
        return left_cols + ["…"] + right_cols
    return cols


def _format_metric_cell(data: dict, col: str, metric: str, stderr_metric: str) -> str:
    if col == "…":
        return "…"
    metrics = data.get(col)
    if not metrics:
        return "-"
    val = metrics.get(metric)
    if isinstance(val, dict):
        val = val.get("value", None)
    if val is not None:
        return "%.2f" % val
    return "-"


def _get_huggingface_providers(model_id: str):
    model_id = model_id.replace("hf-inference-providers/", "").replace(":all", "")
    url = f"https://huggingface.co/api/models/{model_id}"
    params = {"expand[]": "inferenceProviderMapping"}
    response = requests.get(url, params=params)
    response.raise_for_status()  # raise exception for HTTP errors
    data = response.json()
    # Extract provider mapping if available
    providers = data.get("inferenceProviderMapping", {})

    live_providers = []
    for provider, info in providers.items():
        if info.get("status") == "live":
            live_providers.append(provider)

    return live_providers


HELP_PANEL_NAME_1 = "Modeling Parameters"
HELP_PANEL_NAME_2 = "Task Parameters"
HELP_PANEL_NAME_3 = "Connection and parallelization parameters"
HELP_PANEL_NAME_4 = "Logging parameters"


def eval(  # noqa C901
    models: Annotated[list[str], Argument(help="Models to evaluate")],
    tasks: Annotated[str, Argument(help="Tasks to evaluate")],
    # model arguments
    model_base_url: Annotated[
        str | None,
        Option(
            help="Base URL for communicating with the model API.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    model_roles: Annotated[
        str | None,
        Option(
            help="Model creation args (as a dictionary or as a path to a JSON or YAML config file)",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        Option(
            help="The maximum number of tokens that can be generated in the completion (default is model specific)",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    system_message: Annotated[
        str | None,
        Option(
            help="System message to use, overrides the task defined system and default one.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        Option(
            help="Controls randomness in the model's output. Lower values make the model more deterministic and focused, while higher values make it more creative and varied.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    top_p: Annotated[
        float | None,
        Option(
            help="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    top_k: Annotated[
        int | None,
        Option(
            help="The number of highest probability vocabulary tokens to keep for each step of decoding.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    frequence_penalty: Annotated[
        float | None,
        Option(
            help="Number between -2.0 and 2.0, Penalizes tokens that appear in the text too frequently, reducing repetition.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    presence_penalty: Annotated[
        float | None,
        Option(
            help="Number between -2.0 and 2.0, Penalizes tokens that appear in the text, increasing diversity of the generated text.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    logit_bias: Annotated[
        str | None,
        Option(
            help="Bias for each token, can be used to prioritize or deprioritize certain tokens, for example 10=100, -10=-100.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    seed: Annotated[
        int | None, Option(help="Random seed to use for reproducibility", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    stop_seqs: Annotated[
        list[str] | None,
        Option(
            help="Stop sequences to use, can be used to stop the generation of the text.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    num_choices: Annotated[
        int | None,
        Option(help="The number of choices to generate for each step of decoding.", rich_help_panel=HELP_PANEL_NAME_1),
    ] = None,
    best_of: Annotated[
        int | None,
        Option(
            help="Generates best_of completions server-side and returns the one with the highest log probability per token.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    log_probs: Annotated[
        bool | None,
        Option(
            help="Returns log probabilities for each token in the generated text", rich_help_panel=HELP_PANEL_NAME_1
        ),
    ] = None,
    top_logprobs: Annotated[
        int | None,
        Option(help="How many most likely tokens to return at each forward step.", rich_help_panel=HELP_PANEL_NAME_1),
    ] = None,
    cache_prompt: Annotated[
        bool | None, Option(help="Cache prompt prefix.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    reasoning_effort: Annotated[
        str | None, Option(help="Value: `minimal`, `low`, `medium`, `high`", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    reasoning_tokens: Annotated[
        int | None,
        Option(help="Maximum number of tokens to generate for reasoning", rich_help_panel=HELP_PANEL_NAME_1),
    ] = None,
    reasoning_history: Annotated[
        bool | None,
        Option(
            help="values: `none`, `all`, `last`, `auto`. Include reasoning in chat message history sent to generate (defaults to “auto”, which uses the recommended default for each provider)",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ] = None,
    response_format: Annotated[
        str | None, Option(help="JSON schema for the response", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    parallel_tool_calls: Annotated[
        bool | None, Option(help="Enable parallel tool calls", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    max_tool_output: Annotated[
        int | None,
        Option(help="Maximum number of tokens to generate for tool output", rich_help_panel=HELP_PANEL_NAME_1),
    ] = None,
    internal_tools: Annotated[
        bool | None, Option(help="Enable internal tools", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    model_args: Annotated[
        str | None, Option(help="Provider specific arguments: example: 'device=1'", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    # task parameters
    custom_tasks: Annotated[
        str | None,
        Option(
            help="Path to a Python file containing custom task definitions. The file should define a TASKS_TABLE with LightevalTaskConfig objects.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = None,
    max_samples: Annotated[
        int | None,
        Option(
            help="Maximum number of samples to use per task. If your task has multiple subtasks, this will be the maximum number of samples to use per subtask.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = None,
    # Metric parameters
    epochs: Annotated[
        int,
        Option(
            help="Number of times to evaluate the model on the task, the results will be aggregated by the specified reducer.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = 1,
    epochs_reducer: Annotated[
        str | None,
        Option(
            help="Epochs Reducer to use: mean, median, mode, max, at_least_{n}, pass_at_{k}",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = None,
    # Connection and parallelization parameters
    max_connections: Annotated[
        int,
        Option(
            help="Maximum number of concurrent connections to use for each model", rich_help_panel=HELP_PANEL_NAME_3
        ),
    ] = 50,
    timeout: Annotated[
        int, Option(help="Timeout in seconds for each connection", rich_help_panel=HELP_PANEL_NAME_3)
    ] = 30,
    retry_on_error: Annotated[
        int, Option(help="Number of times to retry on error", rich_help_panel=HELP_PANEL_NAME_3)
    ] = 1,
    max_retries: Annotated[
        int, Option(help="Maximum number of retries to use", rich_help_panel=HELP_PANEL_NAME_3)
    ] = 5,
    max_tasks: Annotated[
        int | None, Option(help="Maximum number of tasks to evaluate in parallel", rich_help_panel=HELP_PANEL_NAME_3)
    ] = None,
    # Logging parameters
    log_dir: Annotated[
        str | None,
        Option(help="Log directory to use, will be created if it doesn't exist", rich_help_panel=HELP_PANEL_NAME_4),
    ] = None,
    log_dir_allow_dirty: Annotated[
        bool, Option(help="Allow dirty log directory", rich_help_panel=HELP_PANEL_NAME_4)
    ] = True,
    display: Annotated[
        Literal["rich", "full", "conversations", "plain", "log", "none"],
        Option(help="Display", rich_help_panel=HELP_PANEL_NAME_4),
    ] = "rich",
    bundle_dir: Annotated[
        str | None,
        Option(help="Bundle directory to use, will be created if it doesn't exist", rich_help_panel=HELP_PANEL_NAME_4),
    ] = None,
    bundle_overwrite: Annotated[
        bool,
        Option(help="Overwrite bundle directory if it exists", rich_help_panel=HELP_PANEL_NAME_4),
    ] = True,
    repo_id: Annotated[
        str | None,
        Option(help="Repository ID to use, will be created if it doesn't exist", rich_help_panel=HELP_PANEL_NAME_4),
    ] = None,
    public: Annotated[
        bool,
        Option(
            help="Whether to make the uploaded results and details public on the Hugging Face Hub. If False, datasets will be private.",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = False,
):
    from lighteval.tasks.registry import Registry

    registry = Registry(tasks=tasks, custom_tasks=None, load_multilingual=False)
    task_configs = registry.task_to_configs
    inspect_ai_tasks = []

    for task_name, task_configs in task_configs.items():
        for task_config in task_configs:
            inspect_ai_tasks.append(get_inspect_ai_task(task_config, epochs=epochs, epochs_reducer=epochs_reducer))

    if model_args is not None:
        model_args = InspectAIModelConfig._parse_args(model_args)
    else:
        model_args = {}

    for model in models:
        if model.split("/")[0] == "hf-inference-providers" and model.split(":")[-1] == "all":
            providers = _get_huggingface_providers(model)
            models = [f"{model.replace(':all', '')}:{provider}" for provider in providers]

    if log_dir is None:
        log_dir = f"lighteval-logs-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    success, logs = inspect_ai_eval_set(
        inspect_ai_tasks,
        model=models,
        max_connections=max_connections,
        timeout=timeout,
        retry_on_error=retry_on_error,
        max_retries=max_retries,
        limit=max_samples,
        max_tasks=max_tasks,
        log_dir=log_dir,
        log_dir_allow_dirty=log_dir_allow_dirty,
        display=display,
        model_args=model_args,
        max_tokens=max_tokens,
        system_message=system_message,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequence_penalty=frequence_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        stop_seqs=stop_seqs,
        num_choices=num_choices,
        best_of=best_of,
        log_probs=log_probs,
        top_logprobs=top_logprobs,
        cache_prompt=cache_prompt,
        reasoning_effort=reasoning_effort,
        reasoning_tokens=reasoning_tokens,
        reasoning_history=reasoning_history,
        response_format=response_format,
        parallel_tool_calls=parallel_tool_calls,
        max_tool_output=max_tool_output,
        internal_tools=internal_tools,
        bundle_dir=bundle_dir,
        bundle_overwrite=bundle_overwrite,
    )

    if not success:
        print("Error evaluating models")
        print(f"run the same command with --log-dir {log_dir} to retry !")
        return

    results_per_model_per_task = {}

    for model in models:
        results_per_model_per_task[model] = {}

        for log in logs:
            if log.eval.model == model:
                results_per_model_per_task[model][log.eval.task] = log.results.metrics

    results_per_model_per_task_agg = mean_metrics_by_prefix(results_per_model_per_task)
    table_md = results_to_markdown_table(results_per_model_per_task_agg)

    if repo_id is not None:
        if bundle_dir is not None:
            push_to_hub(bundle_dir, repo_id, public=public)

    print()
    print(table_md)
    print(f"results saved to {log_dir}")

    if log_dir is not None:
        print(f'run "inspect view --log-dir {log_dir}" to view the results')
    else:
        print("run 'inspect view' to view the results")


def bundle(log_dir: str, output_dir: str, overwrite: bool = True, repo_id: str | None = None, public: bool = False):
    bundle_log_dir(log_dir=log_dir, output_dir=output_dir, overwrite=overwrite)

    if repo_id is not None:
        push_to_hub(output_dir, repo_id, public=public)


if __name__ == "__main__":
    tasks = [
        "gsm8k",
        "agieval",
        "hle",
        "ifeval",
        "gpqa",
        "ifbench_test",
        "mmlu_pro",
        "mixeval",
        "aimo",
        "anli",
        "arc",
        "arithmetic",
        "asdiv",
        "babi_qa",
        "bbq",
        "bigbench",
        "bigbench_hard",
        "blimp",
        "bold",
        "boolq",
        "civil_comments",
        "commonsenseqa",
        "covid_dialog",
        "dyck_language",
        "math_500",
        "musr",
        "olympiad_bench",
        "simpleqa",
        "tiny_benchmarks",
    ]
    model = "hf-inference-providers/meta-llama/Llama-3.1-8B-Instruct:nebius"
    eval(models=[model], tasks=tasks[0])
