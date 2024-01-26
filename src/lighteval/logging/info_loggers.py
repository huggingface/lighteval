import collections
import os
import time
from dataclasses import asdict, dataclass, field

import git
import numpy as np
import xxhash

from lighteval.logging.hierarchical_logger import hlog, hlog_warn
from lighteval.metrics import MetricCategory
from lighteval.metrics.stderr import get_stderr_function
from lighteval.models.model_loader import ModelInfo
from lighteval.models.model_output import ModelReturn
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc
from lighteval.utils import as_list, is_nanotron_available, sanitize_numpy


if is_nanotron_available():
    from brrr.config import BrrrConfig


@dataclass(init=False)
class GeneralConfigLogger:
    """Logger for the evaluation parameters.

    Attributes:
        lighteval_sha (str): Current commit sha of lighteval used for the evaluation (for reproducibility purposes)
        num_fewshot_seeds (int): Number of seeds for the few-shot sampling.
            If equal to or below 1, the experiment is done once only, with a single few-shot seed (equal to 0).
            If above, the experiment is reproduced several times, with a different sampling/shuffling for the few-shot examples, which follows what is done in HELM for example.
        override_batch_size (int): Manages the batch size.
            If strictly positive, its value is used as the batch size for all experiments.
            Else, the batch size is automatically inferred depending on what fits in memory.
        max_samples (int): If set, cuts the number of samples per task to `max_samples`.
            Note: This should only be used for debugging purposes!
        job_id (int): If the evaluation suite is launched as a slurm job, stores the current job id.
            Purely informative parameter used to retrieve scheduler logs.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling [`GeneralConfigLogger.log_end_time`]
        total_evaluation_time_secondes (str): Inferred total evaluation time in seconds (from the start and end times).
        model_name (str): Name of the currently evaluated model.
        model_sha (str): Commit hash of the currently evaluated model on the hub if available.
        model_dtype (str): Dtype of the model weights, as obtained when loading the model config.
        model_size (str): Model size as obtained when loading the model config.

    """

    # general
    lighteval_sha: str = None
    num_fewshot_seeds: int = None
    override_batch_size: int = None
    max_samples: int = None
    job_id: int = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_secondes: str = None

    # model info
    model_name: str = None
    model_sha: str = None
    model_dtype: str = None
    model_size: str = None

    # Nanotron/Brrr config
    config: "BrrrConfig" = None

    def __init__(self) -> None:
        """Stores the current lighteval commit for reproducibility, and starts the evaluation timer."""
        repo = git.Repo(os.path.dirname(__file__).split("src")[0])
        self.lighteval_sha = repo.git.rev_parse("HEAD")
        self.start_time = time.perf_counter()

    def log_args_info(self, num_fewshot_seeds, override_batch_size, max_samples, job_id, config=None) -> None:
        self.num_fewshot_seeds = num_fewshot_seeds
        self.override_batch_size = override_batch_size
        self.max_samples = max_samples
        self.job_id = job_id
        self.config = config

    def log_model_info(self, model_info: ModelInfo) -> None:
        self.model_name = model_info.model_name
        self.model_sha = model_info.model_sha
        self.model_dtype = model_info.model_dtype
        self.model_size = model_info.model_size

    def log_end_time(self) -> None:
        self.end_time = time.perf_counter()
        self.total_evaluation_time_secondes = str(self.end_time - self.start_time)


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
            example (str): Current task example query
            instruction (str): Instruction prepended to the example and few shots.
                For example "In this task, you are given information of type x. You need to predict y."
            full_prompt (str): Expanded full prompt (instruction if present, then prompt)
            num_effective_few_shots (int): Number of actual few shots used for the example.
                This depends on the model context length and few-shots samples size: when using effective few-shots,
                only `num_effective_few_shots` few-shot samples are kept, allowing
                1) each of the used few-shot examples and the prompt to not be truncated
                2) this context still allows the model to predict up to the requested max numbers of tokens within its remaining context size.
            num_asked_few_shots (int): Initially asked number of few-shot samples.
            predictions (list): List of the actual model predictions
            input_tokens (list): List of the input tokens given to the model
            cont_tokens (list): List of the continuation tokens predicted by the model
            truncated (list): Size of the truncations (if it was needed to fit the prompt in the model context length)
            padded (list): Size of the padding (if it was needed for the current example)
            gold (list): Example gold targets (for generative evaluations)
            pred_logits (list): List of the actual model predicted logits
            choices (list): List of the possible choices (for multichoice/loglikelihood evaluations)
            gold_index (list): Indices of the gold targets among the [`choices`]
            metrics (dict): Metric name to current example score
        """

        example: str = ""
        instruction: str = ""
        full_prompt: str = ""
        num_effective_few_shots: int = 0
        num_asked_few_shots: int = 0
        predictions: list = field(default_factory=list)
        input_tokens: list = field(default_factory=list)
        cont_tokens: list = field(default_factory=list)
        truncated: list = field(default_factory=list)
        padded: list = field(default_factory=list)
        gold: list = field(default_factory=list)
        pred_logits: list = field(default_factory=list)
        choices: list = field(default_factory=list)
        gold_index: list = field(default_factory=list)
        metrics: dict = field(default_factory=dict)

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
            effective_few_shots (float): Average effective few shots across all samples for the current task.
                The effective few shot is the number of few shots actually used to fit the prompt in the model context
                length while allowing model generation of the expected size.
            num_truncated_few_shots (int): Total number of samples which required truncated prompts to fit the model size for the current task.
        """

        hashes: dict = field(default_factory=dict)
        truncated: int = 0
        non_truncated: int = 0
        padded: int = 0
        non_padded: int = 0
        effective_few_shots: float = 0
        num_truncated_few_shots: int = 0

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
            effective_few_shots (float): Average effective few shots across all samples across all tasks.
                The effective few shot is the number of few shots actually used to fit the prompt in the model context
                length while allowing model generation of the expected size.
            num_truncated_few_shots (int): Number of samples which required truncated prompts to fit the model size across all tasks.
        """

        hashes: dict = field(default_factory=dict)
        truncated: int = 0
        non_truncated: int = 0
        padded: int = 0
        non_padded: int = 0
        num_truncated_few_shots: int = 0

    @dataclass
    class Hash:
        """
        Hashes important values for one sample ([`Doc`]) of one task ([`LightevalTask`])

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
        """
        Hashes the aggregated hash values for all the sample ([`Doc`]) of one task ([`LightevalTask`])

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

    hashes: dict[str, list[Hash]] = collections.defaultdict(list)
    compiled_hashes: dict[str, CompiledHash] = collections.defaultdict(CompiledHash)

    # dict of details for each task, i.e. winogrande: [example1_details, example2_details, ...]
    details: dict[str, list[Detail]] = collections.defaultdict(list)
    compiled_details: dict[str, CompiledDetail] = collections.defaultdict(CompiledDetail)
    compiled_details_over_all_tasks: CompiledDetailOverAllTasks = CompiledDetailOverAllTasks()

    def log(self, task_name: str, task: LightevalTask, doc: Doc, outputs: list[ModelReturn], metrics: dict) -> None:
        """Stores the relevant information for one sample of one task to the total list of samples stored in the DetailsLogger.

        Args:
            task_name (str): Name of the current task of interest.
            task (LightevalTask): Current task of interest.
            doc (Doc): Current sample that we want to store.
            outputs (list[ModelReturn]): Model outputs for the current sample
            metrics (_type_): Model scores for said sample on the current task's metrics.
        """
        detail = self.Detail()
        detail.example = doc.query
        detail.instruction = doc.instruction
        detail.full_prompt = doc.ctx

        predictions = [model_response.get_result_for_eval() for model_response in outputs]

        if isinstance(predictions[0], list):
            # loglikelihood_single_token returns a list of list of floats (but has
            # only one request), we therefore need to flatten the responses in this case.
            predictions = [x for resp in predictions for x in resp]

        detail.predictions = predictions
        detail.input_tokens = [o.input_tokens for o in outputs]
        detail.cont_tokens = [o.generated_tokens for o in outputs]
        detail.truncated = [o.truncated_tokens_count for o in outputs]
        detail.padded = [o.padded_tokens_count for o in outputs]
        detail.num_effective_few_shots = doc.num_effective_few_shots
        detail.num_asked_few_shots = doc.num_asked_few_shots

        pred_saved = False
        if (
            task.has_metric_category[MetricCategory.PERPLEXITY]
            or task.has_metric_category[MetricCategory.TARGET_PERPLEXITY]
        ):
            pred_saved = True
            pass  # should we log something?
        if task.has_metric_category[MetricCategory.GENERATIVE]:
            detail.gold = doc.get_golds()
            pred_saved = True
        if task.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]:
            detail.gold = doc.get_golds()
            detail.pred_logits = [o.logits for o in outputs]
            pred_saved = True
        if task.has_metric_category[MetricCategory.MULTICHOICE]:
            detail.choices = doc.choices
            detail.gold_index = as_list(doc.gold_index)
            pred_saved = True
        if task.has_metric_category[MetricCategory.MULTICHOICE_ONE_TOKEN]:
            detail.choices = doc.choices
            detail.gold_index = as_list(doc.gold_index)
            pred_saved = True
        if not pred_saved:
            raise NotImplementedError(
                "No metric prediction saved."
            )  # We probably need to handle this case if we're here.

        detail.metrics = sanitize_numpy(metrics)
        self.details[task_name].append(detail)

        hash = self.Hash()
        hash.example = xxhash.xxh64(doc.query).hexdigest()
        hash.full_prompt = xxhash.xxh64(doc.ctx).hexdigest()
        hash.input_tokens = xxhash.xxh64(str([o.input_tokens for o in outputs])).hexdigest()
        hash.cont_tokens = xxhash.xxh64(str([o.generated_tokens for o in outputs])).hexdigest()
        self.hashes[task_name].append(hash)

    def aggregate(self):
        """
        Aggregate the details and hashes for each task and then for all tasks.
        We end up with a dict of compiled details for each task and a dict of compiled details for all tasks.
        """

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

        for task_name, task_examples in self.details.items():
            self.compiled_details[task_name].hashes = asdict(self.compiled_hashes[task_name])
            self.compiled_details[task_name].truncated = sum(di > 0 for d in task_examples for di in d.truncated)
            self.compiled_details[task_name].non_truncated = (
                len(task_examples) - self.compiled_details[task_name].truncated
            )
            self.compiled_details[task_name].padded = sum(di > 0 for d in task_examples for di in d.padded)
            self.compiled_details[task_name].non_padded = sum(di == 0 for d in task_examples for di in d.padded)
            self.compiled_details[task_name].effective_few_shots = np.mean(
                [d.num_effective_few_shots for d in task_examples]
            )
            self.compiled_details[task_name].num_truncated_few_shots = sum(
                d.num_effective_few_shots != d.num_asked_few_shots for d in task_examples
            )

        hash_types: list[str] = list(self.compiled_details.values())[0].hashes.keys()

        for hash_type in hash_types:
            self.compiled_details_over_all_tasks.hashes[hash_type] = xxhash.xxh64(
                "".join(
                    compiled_detail.hashes[hash_type] for _, compiled_detail in sorted(self.compiled_details.items())
                )
            ).hexdigest()

        self.compiled_details_over_all_tasks.truncated = sum(d.truncated for d in self.compiled_details.values())
        self.compiled_details_over_all_tasks.non_truncated = sum(
            d.non_truncated for d in self.compiled_details.values()
        )
        self.compiled_details_over_all_tasks.padded = sum(d.padded for d in self.compiled_details.values())
        self.compiled_details_over_all_tasks.non_padded = sum(d.non_padded for d in self.compiled_details.values())
        self.compiled_details_over_all_tasks.num_truncated_few_shots = sum(
            d.num_truncated_few_shots for d in self.compiled_details.values()
        )


class MetricsLogger:
    """Logs the actual scores for each metric of each task.

    Attributes:
        metrics_value (dict[str, dict[str, list[float]]]): Maps each task to its dictionary of metrics to scores for all the example of the task.
            Example: {"winogrande|winogrande_xl": {"accuracy": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}}
        metric_aggregated (dict[str, dict[str, float]]): Maps each task to its dictionary of metrics to aggregated scores over all the example of the task.
            Example: {"winogrande|winogrande_xl": {"accuracy": 0.5}}
    """

    metrics_values: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    metric_aggregated: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(dict))

    def log(self, task_name: str, metrics: dict) -> None:
        for metric_name, metric_value in metrics.items():
            self.metrics_values[task_name][metric_name].append(metric_value)

    def aggregate(self, task_dict: dict[str, LightevalTask], bootstrap_iters: int = 1000):  # noqa: C901
        """
        Aggregate the metrics for each task and then for all tasks.

        Args:
            task_dict (dict[str, LightevalTask]): used to determine what aggregation function to use for each metric
            bootstrap_iters (int, optional): _description_. Defaults to 1000.
        """

        for task_name, metrics in self.metrics_values.items():
            cur_task_name, _ = task_name.rsplit("|", 1)
            # fix the fact that we need the task_dict
            task = task_dict[cur_task_name]

            for metric_name, metric_values in metrics.items():
                try:
                    metric_result = task.aggregation()[metric_name](metric_values)
                except OverflowError:
                    hlog(f"{task_name} {metric_name} OVERFLOW ERROR")
                    metric_result = float("nan")

                if isinstance(metric_result, dict):  # in which cases do we get a dict here?
                    self.metric_aggregated[task_name].update(metric_result)
                else:
                    self.metric_aggregated[task_name][metric_name] = metric_result

                aggregation = task.aggregation()[metric_name]

                stderr = get_stderr_function(aggregation=aggregation, number_experiments=1000)
                if stderr is not None and len(metric_values) > 1:
                    try:
                        self.metric_aggregated[task_name][f"{metric_name}_stderr"] = stderr(metric_values)
                    except OverflowError:
                        # Is this need or should we just pass?
                        self.metric_aggregated[task_name][f"{metric_name}_stderr"] = float("nan")
                        hlog_warn(f"{task_name}, {metric_name} got an OVERFLOW ERROR when computing stderr.")

        suite_average = {}
        suite_nb = {}

        for _, metrics in self.metric_aggregated.items():
            for metric, value in metrics.items():
                suite_average[metric] = suite_average.get(metric, 0) + value
                suite_nb[metric] = suite_nb.get(metric, 0) + 1

        for metric, value in suite_average.items():
            suite_average[metric] = value / suite_nb[metric]

        self.metric_aggregated["all"] = suite_average


class VersionsLogger:
    """Logger of the tasks versions.

    Tasks can have a version number/date, which indicates what is the precise metric definition and dataset version used for an evaluation.

    Attributes:
        version (dict[str, int]): Maps the task names with the task versions.
    """

    # the versions dict will be a dict of task_name: task_version
    # {"winogrande|winogrande_xl": 0}
    versions: dict[str, int] = {"all": 0}

    def log(self, task_name: str, task_version: int) -> None:
        self.versions[task_name] = task_version


class TaskConfigLogger:
    """Logs the different parameters of the current [`LightevalTask`] of interest.

    Attributes:
        tasks_config (dict[str, TaskConfig]): Maps each task to its associated [`TaskConfig`]
    """

    @dataclass
    class TaskConfig:
        """Stored configuration of a given [`LightevalTask`].

        Arguments:
            name (str): Short name of the evaluation task.
            suite (list[str]): Evaluation suites to which the task belongs.
            prompt_function (str): Name of the function used to create the [`Doc`] samples from each line of the evaluation dataset.
            hf_repo (str): Path of the hub dataset repository containing the evaluation information.
            hf_subset (str): Subset used for the current task, will be default if none is selected.
            hf_avail_splits (list[str]): All the available splits in the evaluation dataset
            evaluation_splits (list[str]): List of the splits actually used for this evaluation
            few_shots_split (str): Name of the split from which to sample few-shot examples
            few_shots_select (str): Method with which to sample few-shot examples
            generation_size (int): Maximum allowed size of the generation
            metric (list[str]): List of all the metrics for the current task.
            stop_sequence (list[str]): Stop sequence which interrupts the generation for generative metrics.
            original_num_docs (int): Number of documents in the task
            effective_num_docs (int): Number of documents used in a specific evaluation
            truncated_num_docs (bool): Whether less than the total number of documents were used
            output_regex (str)
            frozen (bool)
        """

        name: str
        suite: list[str]
        prompt_function: str
        hf_repo: str
        hf_subset: str
        hf_avail_splits: list[str]
        evaluation_splits: list[str]
        few_shots_split: str
        few_shots_select: str
        generation_size: int
        metric: list[str]
        stop_sequence: list[str]
        output_regex: str
        frozen: bool
        original_num_docs: int = -1
        effective_num_docs: int = -1

    tasks_configs: dict[str, TaskConfig] = {}

    def log(self, task_dict: dict[str, LightevalTask]) -> None:
        self.tasks_configs = {name: task.cfg for name, task in task_dict.items()}

    def log_num_docs(self, task_name: str, original_num_docs: int, effective_num_docs: int) -> None:
        self.tasks_configs[task_name]["original_num_docs"] = original_num_docs
        self.tasks_configs[task_name]["effective_num_docs"] = effective_num_docs
