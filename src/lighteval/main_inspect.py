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
from typing import Literal

from inspect_ai import Epochs, Task, task
from inspect_ai import eval_set as inspect_ai_eval_set
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message
from pytablewriter import MarkdownTableWriter

from lighteval.models.abstract_model import InspectAIModelConfig
from lighteval.tasks.lighteval_task import LightevalTaskConfig


logger = logging.getLogger(__name__)


@task
def get_inspect_ai_task(lighteval_task_config: LightevalTaskConfig) -> Task:
    name = lighteval_task_config.name
    sample_fields = lighteval_task_config.sample_fields

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
    epochs = 1
    epochs_reducer = "mean"

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


def mean_metrics_by_prefix(results_per_model_per_task, sep=":"):
    out = {}
    for model, tasks in results_per_model_per_task.items():
        pref_metrics = defaultdict(lambda: defaultdict(list))
        # Collect both per-task metrics and values for prefix aggregation
        per_model_out = {}
        for task_name, metrics in tasks.items():
            if sep not in task_name:
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
    se = metrics.get(stderr_metric)
    if isinstance(val, dict):
        val = val.get("value", None)
    if isinstance(se, dict):
        se = se.get("value", None)
    if val is not None and se is not None:
        return "%.4f ± %.4f" % (val, se)
    if val is not None:
        return "%.4f" % val
    return "-"


def eval(
    models: list[str],
    tasks: str,
    epochs: int = 1,
    epochs_reducer: Literal["mean", "median", "mode", "max", "at_least_{n}", "ass_at_{k}"] | None = None,
    max_connections: int = 50,
    timeout: int = 30,
    retry_on_error: int = 1,
    max_retries: int = 5,
    log_dir: str = "lighteval-logs",
    log_dir_allow_dirty: bool = True,
    display: Literal["rich", "full", "conversations", "plain", "log", "none"] = "rich",
    model_config: str | None = None,
    max_samples: int | None = None,
    max_tasks: int | None = None,
):
    from lighteval.tasks.registry import Registry

    registry = Registry(tasks=tasks, custom_tasks=None, load_multilingual=False)
    task_configs = registry.task_to_configs
    inspect_ai_tasks = []

    for task_name, task_configs in task_configs.items():
        for task_config in task_configs:
            inspect_ai_tasks.append(get_inspect_ai_task(task_config))

    if model_config is not None and model_config.endswith(".yaml"):
        model_config = InspectAIModelConfig.from_path(model_config).dict()
    elif model_config is not None:
        model_config = InspectAIModelConfig.from_args(model_config).dict()
    else:
        model_config = {}

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
        **model_config,
    )

    if not success:
        return

    results_per_model_per_task = {}

    for model in models:
        results_per_model_per_task[model] = {}

        for log in logs:
            if log.eval.model == model:
                results_per_model_per_task[model][log.eval.task] = log.results.metrics

    results_per_model_per_task = mean_metrics_by_prefix(results_per_model_per_task)
    table_md = results_to_markdown_table(results_per_model_per_task)
    print(table_md)


if __name__ == "__main__":
    task = "lighteval|gsm8k|5,lighteval|gsm8k|1,lighteval|gsm8k|0"
    task = "lighteval|agieval|0"
    task = "lighteval|hle|0"
    task = "lighteval|ifeval|0"
    task = "lighteval|gpqa|0"
    task = "lighteval|ifbench_test|0"
    model = "hf-inference-providers/meta-llama/Llama-3.1-8B-Instruct:nebius"
    eval(task, model)
