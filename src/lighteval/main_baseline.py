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


from lighteval.cli_args import (
    custom_tasks,
    dataset_loading_processes,
    max_samples,
    output_dir,
    tasks,
)


def baseline(
    tasks: tasks.type,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    output_dir: output_dir.type = output_dir.default,
    max_samples: max_samples.type = max_samples.default,
):
    """Compute baselines for given tasks.

    It has been tested with generative and accuracy tasks, but may not work correctly for other task types.

    The baseline is computed as follows:

    - For multiple-choice tasks: It assumes random guessing, so the score is n_correct/number_of_choices.
    - For other metrics: It assigns a score of 0, which may not be appropriate for all task types.

    Note:
        This baseline computation may not be suitable for all task types and should be used with caution.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.requests import SamplingMethod
    from lighteval.utils.utils import as_list

    registry = Registry(tasks=tasks, custom_tasks=custom_tasks)
    tasks_dict: dict[str, LightevalTask] = registry.load_tasks()

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=False,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    evaluation_tracker.general_config_logger.log_model_info(
        model_config=None,
    )
    evaluation_tracker.task_config_logger.log(tasks_dict)

    LightevalTask.load_datasets(tasks_dict, dataset_loading_processes)

    for task_name, task in tasks_dict.items():
        task_docs = list(task.eval_docs())
        n_samples = min(max_samples, len(task_docs)) if max_samples else len(task_docs)

        p_correct_score = [
            len(as_list(task_doc.gold_index)) / len(task_doc.choices) for task_doc in task_docs[:n_samples]
        ]

        metric_results = {
            metric.metric_name: p_correct_score if metric.category in [SamplingMethod.LOGPROBS] else 0
            for metric in task.metrics
        }

        evaluation_tracker.metrics_logger.log(task_name, metric_results)

    evaluation_tracker.metrics_logger.aggregate(task_dict=tasks_dict, bootstrap_iters=1000)
    evaluation_tracker.save()
