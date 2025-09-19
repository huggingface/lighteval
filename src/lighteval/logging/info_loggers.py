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

import collections
import logging
import os
import time
from dataclasses import asdict, dataclass, field

import git
import xxhash

from lighteval.metrics.utils.stderr import get_stderr_function
from lighteval.models.abstract_model import ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_package_available


logger = logging.getLogger(__name__)


if is_package_available("nanotron"):
    pass


@dataclass(init=False)
class GeneralConfigLogger:
    """Tracks general configuration and runtime information for model evaluations.

    This logger captures key configuration parameters, model details, and timing information
    to ensure reproducibility and provide insights into the evaluation process.

    Attributes:
        lighteval_sha (str): Git commit SHA of lighteval used for evaluation, enabling exact version reproducibility.
            Set to "?" if not in a git repository.

        num_fewshot_seeds (int): Number of random seeds used for few-shot example sampling.
            - If <= 1: Single evaluation with seed=0
            - If > 1: Multiple evaluations with different few-shot samplings (HELM-style)

        max_samples (int, optional): Maximum number of samples to evaluate per task.
            Only used for debugging - truncates each task's dataset.

        job_id (int, optional): Slurm job ID if running on a cluster.
            Used to cross-reference with scheduler logs.

        start_time (float): Unix timestamp when evaluation started.
            Automatically set during logger initialization.

        end_time (float): Unix timestamp when evaluation completed.
            Set by calling log_end_time().

        total_evaluation_time_secondes (str): Total runtime in seconds.
            Calculated as end_time - start_time.

        model_config (ModelConfig): Complete model configuration settings.
            Contains model architecture, tokenizer, and generation parameters.

        model_name (str): Name identifier for the evaluated model.
            Extracted from model_config.
    """

    # general
    lighteval_sha: str = None
    num_fewshot_seeds: int = None
    max_samples: int = None
    job_id: int = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_secondes: str = None

    model_config: ModelConfig = None
    model_name: str = None

    def __init__(self) -> None:
        """Stores the current lighteval commit for reproducibility, and starts the evaluation timer."""
        try:
            repo = git.Repo(os.path.dirname(__file__).split("src")[0])
        except git.InvalidGitRepositoryError:
            repo = None

        self.lighteval_sha = repo.git.rev_parse("HEAD") if repo is not None else "?"
        self.start_time = time.perf_counter()

    def log_args_info(
        self,
        num_fewshot_seeds: int,
        max_samples: int | None,
        job_id: str,
    ) -> None:
        """Logs the information about the arguments passed to the method.

        Args:
            num_fewshot_seeds (int): number of few-shot seeds.
            max_samples (int | None): maximum number of samples, if None, use all the samples available.
            job_id (str): job ID, used to retrieve logs.
        """
        self.num_fewshot_seeds = num_fewshot_seeds
        self.max_samples = max_samples
        self.job_id = job_id

    def log_model_info(self, model_config: ModelConfig) -> None:
        """Logs the model information.

        Args:
            model_config: the model config used to initialize the model.
        """
        self.model_config = model_config
        self.model_name = model_config.model_name

    def log_end_time(self) -> None:
        self.end_time = time.perf_counter()
        self.total_evaluation_time_secondes = str(self.end_time - self.start_time)


@dataclass()
class DetailsLogger:
    """Logger for the experiment details.

    Stores and logs experiment information both at the task and at the sample level.

    Attributes:
        hashes (dict[str, list[`Hash`]): Maps each task name to the list of all its samples' [`Hash`].
        compiled_hashes (dict[str, CompiledHash): Maps each task name to its [`CompiledHas`], an aggregation of all the individual sample hashes.
        details (dict[str, list[`Detail`]]): Maps each task name to the list of its samples' details.
            Example: winogrande: [sample1_details, sample2_details, ...]
        compiled_details (dict[str, `CompiledDetail`]): : Maps each task name to the list of its samples' compiled details.
        compiled_details_over_all_tasks (CompiledDetailOverAllTasks): Aggregated details over all the tasks.

    """

    @dataclass()
    class Detail:
        """Experiment details of one single example of one task.

        Attributes:
            doc (Doc): The [`Doc`] object containing the current example information.
            model_response (ModelResponse): The [`ModelResponse`] object containing the model response for the current example.
            metric (dict): The metric scores for the current example.
                Example: {"accuracy": 0.5, "f1": 0.7, "exact_match": 0.6}
        """

        doc: Doc
        model_response: ModelResponse
        metric: dict

    @dataclass
    class CompiledDetail:
        """Experiment details compiled (for all samples) over one evaluation task.

        Attributes:
            hashes (dict): A dictionary version of the [`CompiledHash`] for the current task samples.
                These compiled hashes allow to quickly observe whether there is a difference between the aggregated
                prompts or generations between two evaluation runs for a given task.
            truncated (int): Total number of samples which needed prompt truncation to fit the model context size for the current task.
            non_truncated (int): Total number of samples which did not need prompt truncation to fit the model context size for the current task.
            padded (int): Total umber of samples which needed padding during the batching step for the current task.
            non_padded (int): Total number of samples which did not need padding during the batching step for the current task.
        """

        hashes: dict = field(default_factory=dict)
        truncated: int = 0
        non_truncated: int = 0
        padded: int = 0
        non_padded: int = 0

    @dataclass
    class CompiledDetailOverAllTasks:
        """Experiment details compiled across all evaluation tasks.

        Attributes:
            hashes (dict): For each key, average hash of the [`CompiledHash`] values across all tasks.
                These compiled hashes allow to quickly observe whether there is a difference between the aggregated
                prompts or generations between two evaluation runs across all tasks.
            truncated (int): Total number of samples which needed prompt truncation to fit the model context size across all tasks.
            non_truncated (int): Total number of samples which did not need prompt truncation to fit the model context size across all tasks
            padded (int): Number of samples which needed padding during the batching step across all tasks.
            non_padded (int): Number of samples which did not need padding during the batching step across all tasks.
        """

        hashes: dict = field(default_factory=dict)
        truncated: int = 0
        non_truncated: int = 0
        padded: int = 0
        non_padded: int = 0

    @dataclass
    class Hash:
        """Hashes important values for one sample ([`Doc`]) of one task ([`LightevalTask`])

        Attributes:
            example (str): Hash of the [`Doc.query`]
            full_prompt (str): Hash of the [`Doc.ctx`]
            input_tokens (str): Aggregated hash of all the [`Doc.input_tokens`]
            cont_tokens (str): Aggregated hash of all the [`Doc.generated_tokens`]

        """

        example: str = ""
        full_prompt: str = ""
        input_tokens: str = ""
        cont_tokens: str = ""

    @dataclass
    class CompiledHash:
        """Hashes the aggregated hash values for all the sample ([`Doc`]) of one task ([`LightevalTask`])

        Attributes:
            example (str): Aggregated hash of all the [`Doc.query`] hashes for all samples of the current task.
            full_prompt (str): Aggregated hash of all the [`Doc.ctx`] hashes for all samples of the current task.
            input_tokens (str): Aggregated hash of the aggregated [`Doc.input_tokens`] hashes over all samples of the current task.
            cont_tokens (str): Aggregated hash of the aggregated [`Doc.generated_tokens`] hashes over all samples of the current task.
        """

        hash_examples: str = ""
        hash_full_prompts: str = ""
        hash_input_tokens: str = ""
        hash_cont_tokens: str = ""

    hashes: dict[str, list[Hash]] = field(default_factory=lambda: collections.defaultdict(list))
    compiled_hashes: dict[str, CompiledHash] = field(
        default_factory=lambda: collections.defaultdict(DetailsLogger.CompiledHash)
    )

    # dict of details for each task, i.e. winogrande: [example1_details, example2_details, ...]
    details: dict[str, list[Detail]] = field(default_factory=lambda: collections.defaultdict(list))
    compiled_details: dict[str, CompiledDetail] = field(
        default_factory=lambda: collections.defaultdict(DetailsLogger.CompiledDetail)
    )
    compiled_details_over_all_tasks: CompiledDetailOverAllTasks = field(default_factory=CompiledDetailOverAllTasks)

    def log(
        self,
        task_name: str,
        doc: Doc,
        model_response: ModelResponse,
        metrics: dict,
    ) -> None:
        """Stores the relevant information for one sample of one task to the total list of samples stored in the DetailsLogger.

        Args:
            task_name (str): Name of the current task of interest.
            doc (Doc): Current sample that we want to store.
            model_response (ModelResponse): Model outputs for the current sample
            metrics (dict): Model scores for said sample on the current task's metrics.
        """
        detail = self.Detail(doc, model_response, metrics)
        self.details[task_name].append(detail)

        hash = self.Hash()
        hash.example = xxhash.xxh64(doc.query).hexdigest()
        hash.input_tokens = xxhash.xxh64(str(model_response.input_tokens)).hexdigest()
        hash.cont_tokens = xxhash.xxh64(str(model_response.output_tokens)).hexdigest()
        self.hashes[task_name].append(hash)

    def aggregate(self):
        """Hashes the details for each task and then for all tasks."""
        for task_name in self.hashes:
            compiled_hash = self.CompiledHash()
            compiled_hash.hash_examples = xxhash.xxh64(
                "".join(sorted(q.example for q in self.hashes[task_name]))
            ).hexdigest()  # hash of all the hash - sorted for reproducibility
            compiled_hash.hash_full_prompts = xxhash.xxh64(
                "".join(sorted(q.full_prompt for q in self.hashes[task_name]))
            ).hexdigest()  # hash of all the hash - sorted for reproducibility
            compiled_hash.hash_input_tokens = xxhash.xxh64(
                "".join(sorted(q.input_tokens for q in self.hashes[task_name]))
            ).hexdigest()  # hash of all the hash - sorted for reproducibility
            compiled_hash.hash_cont_tokens = xxhash.xxh64(
                "".join(sorted(q.cont_tokens for q in self.hashes[task_name]))
            ).hexdigest()  # hash of all the hash - sorted for reproducibility
            self.compiled_hashes[task_name] = compiled_hash

        for task_name, _ in self.details.items():
            self.compiled_details[task_name].hashes = asdict(self.compiled_hashes[task_name])

        hash_types: list[str] = list(self.compiled_details.values())[0].hashes.keys()

        for hash_type in hash_types:
            self.compiled_details_over_all_tasks.hashes[hash_type] = xxhash.xxh64(
                "".join(
                    compiled_detail.hashes[hash_type] for _, compiled_detail in sorted(self.compiled_details.items())
                )
            ).hexdigest()


@dataclass
class MetricsLogger:
    """Logs the actual scores for each metric of each task.

    Attributes:
        metrics_value (dict[str, dict[str, list[float]]]): Maps each task to its dictionary of metrics to scores for all the example of the task.
            Example: {"winogrande|winogrande_xl": {"accuracy": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}}
        metric_aggregated (dict[str, dict[str, float]]): Maps each task to its dictionary of metrics to aggregated scores over all the example of the task.
            Example: {"winogrande|winogrande_xl": {"accuracy": 0.5}}
    """

    metrics_values: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    metric_aggregated: dict[str, dict[str, float]] = field(
        default_factory=lambda: collections.defaultdict(lambda: collections.defaultdict(float))
    )

    def log(self, task_name: str, metrics: dict) -> None:
        for metric_name, metric_value in metrics.items():
            self.metrics_values[task_name][metric_name].append(metric_value)

    def aggregate(self, task_dict: dict[str, LightevalTask], bootstrap_iters: int = 1000):  # noqa: C901
        """Aggregate the metrics for each task and then for all tasks.

        Args:
            task_dict (dict[str, LightevalTask]): used to determine what aggregation function to use for each metric
            bootstrap_iters (int, optional): Number of runs used to run the statistical bootstrap. Defaults to 1000.
        """
        for task_name, metrics in self.metrics_values.items():
            task = task_dict[task_name]

            skip_metric = []
            for metric_name, metric_values in metrics.items():
                if metric_name in skip_metric:
                    # The metric is in a subset which has already been computed and saved
                    continue

                aggregation = task.aggregation()[metric_name]

                try:
                    metric_result = aggregation(metric_values)
                except OverflowError:
                    logger.warning(f"{task_name}, {metric_name} got an OVERFLOW ERROR when aggregating.")
                    metric_result = float("nan")

                if isinstance(metric_result, dict):  # For some corpus level grouping metrics
                    self.metric_aggregated[task_name].update(metric_result)
                    skip_metric.extend(list(metric_result.keys()))  # no need to recompute them later
                else:
                    self.metric_aggregated[task_name][metric_name] = metric_result

                if isinstance(metric_result, dict) or bootstrap_iters == 0:
                    stderr = (
                        None  # We skip stderr for some corpus metrics that return dicts, or if bootstrap_iters is 0
                    )
                else:
                    stderr = get_stderr_function(aggregation=aggregation, number_experiments=bootstrap_iters)
                if stderr is not None and len(metric_values) > 1:
                    try:
                        self.metric_aggregated[task_name][f"{metric_name}_stderr"] = stderr(metric_values)
                    except OverflowError:
                        # Is this need or should we just pass?
                        self.metric_aggregated[task_name][f"{metric_name}_stderr"] = float("nan")
                        logger.warning(f"{task_name}, {metric_name} got an OVERFLOW ERROR when computing stderr.")

        # We group subtasks which belong to the same parent task, like MMLU, to compute an average on them
        # and compute an average of all metrics
        grouped_tasks = collections.defaultdict(list)
        suite_average = {}
        suite_nb = {}

        # Build aggregation
        for k, metrics in self.metric_aggregated.items():
            if "|" in k:
                suite, task, fewshot = k.split("|")
                grouped_tasks[f"{suite}|{task.split(':')[0]}:_average|{fewshot}"].append(k)
            for metric, value in metrics.items():
                suite_average[metric] = suite_average.get(metric, 0) + value
                suite_nb[metric] = suite_nb.get(metric, 0) + 1

        # Compute average for sub groups
        for average_task, list_of_subtasks in grouped_tasks.items():
            if len(list_of_subtasks) > 1:
                metrics = list(self.metric_aggregated[list_of_subtasks[0]].keys())
                self.metric_aggregated[average_task] = {
                    metric: sum(self.metric_aggregated[k][metric] for k in list_of_subtasks) / len(list_of_subtasks)
                    for metric in metrics
                }

        # Compute average for all
        for metric, value in suite_average.items():
            suite_average[metric] = value / suite_nb[metric]

        self.metric_aggregated["all"] = suite_average


@dataclass
class VersionsLogger:
    """Logger of the tasks versions.

    Tasks can have a version number/date, which indicates what is the precise metric definition and dataset version used for an evaluation.

    Attributes:
        version (dict[str, int]): Maps the task names with the task versions.

    """

    # the versions dict will be a dict of task_name: task_version
    # {"winogrande|winogrande_xl": 0}
    versions: dict[str, int] = field(default_factory=dict)

    def log(self, task_name: str, task_version: int) -> None:
        self.versions[task_name] = task_version


@dataclass
class TaskConfigLogger:
    """Logs the different parameters of the current [`LightevalTask`] of interest.

    Attributes:
        tasks_config (dict[str, LightevalTaskConfig]): Maps each task to its associated [`LightevalTaskConfig`]

    """

    tasks_configs: dict[str, LightevalTaskConfig] = field(default_factory=dict)

    def log(self, task_dict: dict[str, LightevalTask]) -> None:
        self.tasks_configs = {name: task.config for name, task in task_dict.items()}

    def log_num_docs(self, task_name: str, original_num_docs: int, effective_num_docs: int) -> None:
        self.tasks_configs[task_name].original_num_docs = original_num_docs
        self.tasks_configs[task_name].effective_num_docs = effective_num_docs
