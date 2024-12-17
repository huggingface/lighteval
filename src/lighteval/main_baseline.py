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


import os
from typing import Optional

from typer import Argument, Option
from typing_extensions import Annotated


CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


def baseline(
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = CACHE_DIR,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = "results",
    max_samples: Annotated[
        Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = None,
):
    """
    Compute baselines for given tasks.

    It has been tested with generative and accuracy tasks, but may not work correctly for other task types.

    The baseline is computed as follows:

    - For multiple-choice tasks: It assumes random guessing, so the score is n_correct/number_of_choices.
    - For other metrics: It assigns a score of 0, which may not be appropriate for all task types.

    Note:
        This baseline computation may not be suitable for all task types and should be used with caution.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.metrics.utils.metric_utils import MetricCategory
    from lighteval.models.abstract_model import ModelInfo
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry, taskinfo_selector
    from lighteval.utils.utils import as_list

    task_registry = Registry(cache_dir=cache_dir, custom_tasks=custom_tasks)
    task_names_list, fewshots_dict = taskinfo_selector(tasks, task_registry)
    task_dict = task_registry.get_task_dict(task_names_list)

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=False,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    evaluation_tracker.general_config_logger.log_model_info(
        ModelInfo(
            model_name="lighteval/baseline",
            model_sha=None,
            model_dtype=None,
            model_size=None,
        )
    )
    evaluation_tracker.task_config_logger.log(task_dict)

    LightevalTask.load_datasets(list(task_dict.values()), dataset_loading_processes)

    for task_name, task in task_dict.items():
        task_docs = list(task.eval_docs())
        n_samples = min(max_samples, len(task_docs)) if max_samples else len(task_docs)

        p_correct_score = [
            len(as_list(task_doc.gold_index)) / len(task_doc.choices) for task_doc in task_docs[:n_samples]
        ]

        metric_results = {
            metric.metric_name: p_correct_score
            if metric.category
            in [MetricCategory.MULTICHOICE, MetricCategory.MULTICHOICE_PMI, MetricCategory.MULTICHOICE_ONE_TOKEN]
            else 0
            for metric in task.metrics
        }

        for fewshots, _ in fewshots_dict[task_name]:
            evaluation_tracker.metrics_logger.log(f"{task_name}|{fewshots}", metric_results)

    evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict, bootstrap_iters=1000)
    evaluation_tracker.save()
