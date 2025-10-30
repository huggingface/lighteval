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


from inspect_ai import Epochs, Task, task
from inspect_ai import eval_set as inspect_ai_eval_set
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, system_message

from lighteval.models.abstract_model import InspectAIModelConfig
from lighteval.tasks.lighteval_task import LightevalTaskConfig


@task
def get_inspect_ai_task(lighteval_task_config: LightevalTaskConfig):
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
    epochs = 1
    epochs_reducer = None

    if lighteval_task_config.num_fewshots > 0:
        name += f"_{lighteval_task_config.num_fewshots}_shots"
        # todo: use fewshot split
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


def eval(
    models: list[str],
    tasks: str,
    epochs: int = 1,
    max_connections: int = 50,
    timeout: int = 30,
    retry_on_error: int = 1,
    max_retries: int = 5,
    log_dir: str = "lighteval-logs",
    log_dir_allow_dirty: bool = True,
    display: str = "rich",
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

    inspect_ai_eval_set(
        inspect_ai_tasks,
        model=models,
        max_connections=max_connections,
        timeout=timeout,
        retry_on_error=retry_on_error,
        max_retries=max_retries,
        epochs=epochs,
        limit=max_samples,
        max_tasks=max_tasks,
        log_dir=log_dir,
        log_dir_allow_dirty=log_dir_allow_dirty,
        display=display,
        **model_config,
    )


if __name__ == "__main__":
    task = "lighteval|gsm8k|5,lighteval|gsm8k|1,lighteval|gsm8k|0"
    task = "lighteval|agieval|0"
    task = "lighteval|hle|0"
    task = "lighteval|ifeval|0"
    task = "lighteval|gpqa|0"
    task = "lighteval|ifbench_test|0"
    model = "hf-inference-providers/meta-llama/Llama-3.1-8B-Instruct:nebius"
    eval(task, model)
