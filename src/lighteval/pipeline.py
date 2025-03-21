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

import ast
import collections
import os
import random
import re
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import Enum, auto

import numpy as np
from tqdm import tqdm

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.utils.metric_utils import MetricCategory
from lighteval.models.model_loader import TransformersModel, load_model
from lighteval.models.model_output import (
    GenerativeMultiturnResponse,
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
    ModelResponse,
)
from lighteval.tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from lighteval.tasks.registry import Registry, taskinfo_selector
from lighteval.tasks.requests import RequestType, SampleUid
from lighteval.utils.imports import (
    NO_ACCELERATE_ERROR_MSG,
    NO_NANOTRON_ERROR_MSG,
    NO_OPENAI_ERROR_MSG,
    NO_SGLANG_ERROR_MSG,
    NO_TGI_ERROR_MSG,
    NO_VLLM_ERROR_MSG,
    is_accelerate_available,
    is_nanotron_available,
    is_openai_available,
    is_sglang_available,
    is_tgi_available,
    is_vllm_available,
)
from lighteval.utils.parallelism import test_all_gather
from lighteval.utils.utils import EnvConfig, make_results_table


if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
if is_nanotron_available():
    from nanotron import distributed as dist
    from nanotron.parallel.context import ParallelContext
    from nanotron.utils import local_ranks_zero_first

    from lighteval.models.nanotron_model import NanotronLightevalModel


import logging


logger = logging.getLogger(__name__)


class ParallelismManager(Enum):
    ACCELERATE = auto()
    NANOTRON = auto()
    TGI = auto()
    OPENAI = auto()
    VLLM = auto()
    NONE = auto()
    SGLANG = auto()


@dataclass
class PipelineParameters:
    launcher_type: ParallelismManager
    # Env parameters
    env_config: EnvConfig = field(default_factory=EnvConfig)
    job_id: int = 0
    dataset_loading_processes: int = 1
    nanotron_checkpoint_path: str | None = None  # only for nanotron models
    # Dataset
    custom_tasks_directory: str | None = None
    # Generation parameters
    override_batch_size: int | None = None
    num_fewshot_seeds: int = 1
    max_samples: int | None = None
    use_chat_template: bool = False
    system_prompt: str | None = None
    load_responses_from_details_date_id: str | None = None

    def __post_init__(self):  # noqa C901
        if self.launcher_type == ParallelismManager.ACCELERATE:
            if not is_accelerate_available():
                raise ImportError(NO_ACCELERATE_ERROR_MSG)
        elif self.launcher_type == ParallelismManager.VLLM:
            if not is_vllm_available():
                raise ImportError(NO_VLLM_ERROR_MSG)
        elif self.launcher_type == ParallelismManager.SGLANG:
            if not is_sglang_available():
                raise ImportError(NO_SGLANG_ERROR_MSG)
        elif self.launcher_type == ParallelismManager.TGI:
            if not is_tgi_available():
                raise ImportError(NO_TGI_ERROR_MSG)
        elif self.launcher_type == ParallelismManager.NANOTRON:
            if not is_nanotron_available():
                raise ImportError(NO_NANOTRON_ERROR_MSG)
        elif self.launcher_type == ParallelismManager.OPENAI:
            if not is_openai_available():
                raise ImportError(NO_OPENAI_ERROR_MSG)


class Pipeline:
    def __init__(
        self,
        tasks: str,
        pipeline_parameters: PipelineParameters,
        evaluation_tracker: EvaluationTracker,
        model_config=None,
        model=None,
        metric_options=None,
    ):
        if not (model or model_config):
            raise ValueError("Must provide either a model or model config when creating a pipeline.")

        self.pipeline_parameters = pipeline_parameters
        self.launcher_type = self.pipeline_parameters.launcher_type
        if self.pipeline_parameters.max_samples:
            logger.warning(
                "--max_samples WAS SET. THESE NUMBERS ARE ONLY PARTIAL AND SHOULD NOT BE USED FOR COMPARISON UNLESS YOU KNOW WHAT YOU ARE DOING."
            )

        self.model_config = model_config
        self.evaluation_tracker = evaluation_tracker
        self._metric_options = metric_options or {}
        self.accelerator, self.parallel_context = self._init_parallelism_manager()
        self.model = self._init_model(model_config, model)

        generation_parameters = asdict(model_config.generation_parameters) if model_config else {}

        self.evaluation_tracker.general_config_logger.log_model_info(generation_parameters, self.model.model_info)
        self._init_tasks_and_requests(tasks=tasks)
        self._init_random_seeds()
        # Final results
        self.final_dict: dict = None

    def _init_parallelism_manager(self):
        accelerator, parallel_context = None, None
        if self.launcher_type == ParallelismManager.ACCELERATE:
            if not is_accelerate_available():
                raise ValueError("You are trying to launch an accelerate model, but accelerate is not installed")
            accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
            test_all_gather(accelerator=accelerator)
        elif self.launcher_type == ParallelismManager.NANOTRON:
            if not is_nanotron_available():
                raise ValueError("You are trying to launch a nanotron model, but nanotron is not installed")
            dist.initialize_torch_distributed()
            parallel_context = ParallelContext(
                tensor_parallel_size=self.model_config.lighteval_config.parallelism.tp,
                pipeline_parallel_size=self.model_config.lighteval_config.parallelism.pp,
                data_parallel_size=self.model_config.lighteval_config.parallelism.dp,
            )
            test_all_gather(parallel_context=parallel_context)

        return accelerator, parallel_context

    def _init_model(self, model_config, model):
        logger.info("--- LOADING MODEL ---")
        if model_config is not None:
            if self.parallel_context:
                return NanotronLightevalModel(
                    checkpoint_path=os.path.dirname(self.pipeline_parameters.nanotron_checkpoint_path)
                    if self.pipeline_parameters.nanotron_checkpoint_path
                    else "",
                    nanotron_config=self.model_config,
                    parallel_context=self.parallel_context,
                    debug_one_layer_model=False,
                    model_class=None,
                    env_config=self.pipeline_parameters.env_config,
                )
            else:
                return load_model(config=model_config, env_config=self.pipeline_parameters.env_config)
        if isinstance(model, TransformersModel):
            return model
        else:
            return TransformersModel.from_model(
                model=model,
                use_chat_template=self.pipeline_parameters.use_chat_template,
                env_config=self.pipeline_parameters.env_config,
                accelerator=self.accelerator,
            )

    def _init_tasks_and_requests(self, tasks: str):
        with local_ranks_zero_first() if self.launcher_type == ParallelismManager.NANOTRON else nullcontext():
            logger.info("--- LOADING TASKS ---")
            registry = Registry(
                cache_dir=self.pipeline_parameters.env_config.cache_dir,
                custom_tasks=self.pipeline_parameters.custom_tasks_directory,
            )
            task_names_list, fewshots_dict = taskinfo_selector(tasks, registry)
            task_dict = registry.get_task_dict(task_names_list)
            # If there are metric_options defined from the yaml file,
            # review if they have to be updated.
            if self._metric_options:
                self._update_num_samples(task_dict)
            LightevalTask.load_datasets(list(task_dict.values()), self.pipeline_parameters.dataset_loading_processes)

            self.evaluation_tracker.task_config_logger.log(task_dict)

            requests, docs = create_requests_from_tasks(
                task_dict=task_dict,
                fewshot_dict=fewshots_dict,
                num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
                lm=self.model,
                max_samples=self.pipeline_parameters.max_samples,
                evaluation_tracker=self.evaluation_tracker,
                use_chat_template=self.pipeline_parameters.use_chat_template,
                system_prompt=self.pipeline_parameters.system_prompt,
            )

            self.task_names_list = task_names_list
            self.task_dict = task_dict
            self.fewshot_dict = fewshots_dict
            self.requests = requests
            self.docs = docs

    def _update_num_samples(self, task_dict: dict[str, LightevalTask]):
        """Helper function to update the num_samples of a given metric via the yaml file.
        As it has to be done at the metric level, it's better to update the value per metric.
        It will add a num_samples to the already defined metrics' num_samples if defined in the yaml file.
        As later when constructing the requests the max is taken over the num_samples, this is valid.
        """
        for _, task in task_dict.items():
            for metric in task.metrics:
                if metric_data := self._metric_options.get(metric.metric_name, None):
                    num_samples = metric_data.get("num_samples", None)
                    if num_samples:
                        task.num_samples = [num_samples]

    def _init_random_seeds(self):
        logger.info("--- INIT SEEDS ---")
        random.seed(1234)
        np.random.seed(1234)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        if self.parallel_context is not None:
            dist.barrier()

    def is_main_process(self):
        if self.accelerator:
            return self.accelerator.is_main_process
        if self.parallel_context:
            return dist.get_rank(self.parallel_context.world_pg) == 0
        return True

    def evaluate(self):
        self.evaluation_tracker.general_config_logger.log_args_info(
            num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
            override_batch_size=self.pipeline_parameters.override_batch_size,
            max_samples=self.pipeline_parameters.max_samples,
            job_id=self.pipeline_parameters.job_id,
            config=self.model_config,
        )

        if self.pipeline_parameters.load_responses_from_details_date_id:
            try:
                sample_id_to_responses = self._load_responses_from_details()
            except FileNotFoundError as e:
                logger.warning(
                    f"No responses found for {self.pipeline_parameters.load_responses_from_details_date_id} in details directory: {e}. Running model instead."
                )
                sample_id_to_responses = self._run_model()
        else:
            sample_id_to_responses = self._run_model()

        self._compute_metrics(sample_id_to_responses)

        if self.is_main_process():
            self.evaluation_tracker.general_config_logger.log_end_time()
            self.evaluation_tracker.metrics_logger.aggregate(task_dict=self.task_dict, bootstrap_iters=1000)
            self.evaluation_tracker.details_logger.aggregate()

            for weights in ["delta", "adapter"]:
                try:
                    tmp_weights_dir = f"{self.evaluation_tracker.general_config_logger.model_name}-{weights}-applied"
                    shutil.rmtree(tmp_weights_dir)
                    logger.info(f"Removed {tmp_weights_dir}")
                except OSError:
                    pass

    def _unpack(self, x):
        if isinstance(x, str):
            return x
        elif isinstance(x, (list, tuple)):
            return self._unpack(x[0])
        else:
            raise ValueError(f"Unknown type {type(x)} of prediction {x}")

    def _parse_tensor_string(self, tensor_string):
        """
        Convert a string containing PyTorch-like `tensor([...], device='cuda:0', ...)`
        into a Python list (or nested lists) of numbers.

        Example:
            "[tensor([1, 2, 3], device='cuda:0'), tensor([[4,5],[6,7]], dtype=torch.int64)]"
            -> [[1, 2, 3], [[4, 5], [6, 7]]]
        """

        # Regex explanation:
        #   - tensor\(\s*: Matches "tensor(" (possibly with spaces after), literally.
        #   - (.*?): Captures everything lazily into group(1), until the first subsequent part matches.
        #     We rely on the next pattern to anchor the end of this capture.
        #   - \): The literal closing parenthesis, but we anchor the match by ignoring
        #     further arguments (device=..., dtype=..., etc.) inside.
        #
        #   The tricky part: a tensor might look like
        #   tensor([ ... ], device='cuda:0', dtype=torch.int64)
        #   so the bracket portion is `[ ... ]`, but it can have newlines, etc.
        #
        #   We'll handle that by first capturing the entire content up to the final parenthesis,
        #   then parse out the bracket portion. This can be done in a function-based re.sub.

        pattern = re.compile(
            r"tensor\s*\(\s*(.*?)\s*\)",  # capture everything inside tensor(...)
            flags=re.DOTALL,
        )

        def tensor_replacer(match):
            inside = match.group(1).strip()
            # `inside` might look like: [1, 2, 3], device='cuda:0'
            # or:
            #   [
            #     1, 2, 3,
            #     4, 5, ...
            #   ], device='cuda:0', dtype=torch.int64
            #
            # 1) Extract the bracketed array portion: the first [ ... ] block
            #    which might be multi-line. We'll use another regex for that.

            # We look for the bracketed portion from the first '[' to its matching ']'.
            # Because the inside can be multi-line, we use DOTALL. But we still need
            # to ensure we don't accidentally go beyond the matching bracket.
            #
            # A robust approach to properly match brackets can be done with a small parser,
            # but for typical well-formed strings, a lazy match of the form
            # r"\[.*?\]" DOTALL often suffices, assuming no nested brackets inside.

            bracket_pattern = re.compile(r"\[.*?\]", re.DOTALL)
            bracket_match = bracket_pattern.search(inside)
            if not bracket_match:
                # If we fail to find a bracket, just return something safe.
                # This means the string didn't match the expected format.
                return "[]"

            # The bracketed portion (e.g. "[1, 2, 3\n, 4]").
            bracketed_content = bracket_match.group(0)

            # Return just the bracketed content,
            # effectively replacing "tensor(...)" with "[...]".
            return bracketed_content

        # Step 1: Replace every `tensor(...)` occurrence with just the bracketed list.
        processed = pattern.sub(tensor_replacer, tensor_string)

        # Step 2: Now we can safely parse the result with literal_eval.
        #         If there's still something weird, it may throw ValueError.
        try:
            return ast.literal_eval(processed)
        except Exception as e:
            raise ValueError(f"Failed to parse after preprocessing. " f"Processed string:\n{processed}\n\nError: {e}")

    def _load_responses_from_details(self):
        logger.info("--- LOADING RESPONSES FROM DETAILS ---")
        sample_id_to_responses: dict[(SampleUid, MetricCategory), list[ModelResponse]] = collections.defaultdict(list)

        request_types = list(self.requests.keys())
        if len(request_types) > 1:
            raise ValueError(
                "Loading responses from details when there are multiple request types is currently not supported"
            )
        model_response_type = self._get_model_response_type(request_types[0])

        details_datasets = self.evaluation_tracker.load_details_datasets(
            self.pipeline_parameters.load_responses_from_details_date_id, self.task_names_list
        )

        for task_name, dataset in tqdm(details_datasets.items(), desc="Loading responses from details for tasks"):
            task: LightevalTask = self._get_task(task_name)
            num_samples = len(set(dataset["specifics"]))
            max_samples = self.pipeline_parameters.max_samples if self.pipeline_parameters.max_samples else num_samples
            if num_samples > max_samples:
                logger.warning(
                    f"Skipping {num_samples - max_samples} samples for {task_name} when loading responses from details because max_samples is set to {max_samples}"
                )
                num_samples = self.pipeline_parameters.max_samples

            predictions = [self._unpack(ast.literal_eval(p)) for p in dataset["predictions"][:num_samples]]
            input_tokens = [self._parse_tensor_string(t) for t in dataset["input_tokens"][:num_samples]]
            cont_tokens = [self._parse_tensor_string(t) for t in dataset["cont_tokens"][:num_samples]]
            truncated = [ast.literal_eval(t)[0] for t in dataset["truncated"][:num_samples]]
            padded = [ast.literal_eval(p)[0] for p in dataset["padded"][:num_samples]]

            if model_response_type == GenerativeResponse:
                logits = [ast.literal_eval(p) for p in dataset["pred_logits"][:num_samples]]

            for metric_category, has_metric_category in task.has_metric_category.items():
                if not has_metric_category:
                    continue

                for idx in range(num_samples):
                    kwargs = {
                        "result": predictions[idx],
                        "input_tokens": input_tokens[idx],
                        "generated_tokens": cont_tokens[idx],
                        "truncated_tokens_count": truncated[idx],
                        "padded_tokens_count": padded[idx],
                    }
                    if model_response_type == GenerativeResponse:
                        kwargs["logits"] = logits[idx]

                    response = model_response_type(**kwargs)
                    sample_id_to_responses[(SampleUid(task_name, f"{idx}_{0}"), metric_category)] = [response]
        return sample_id_to_responses

    def _get_model_response_type(self, request_type):
        if request_type == RequestType.LOGLIKELIHOOD:
            model_response_type = LoglikelihoodResponse
        elif request_type == RequestType.LOGLIKELIHOOD_SINGLE_TOKEN:
            model_response_type = LoglikelihoodSingleTokenResponse
        elif request_type == RequestType.LOGLIKELIHOOD_ROLLING:
            model_response_type = LoglikelihoodResponse
        elif request_type == RequestType.GREEDY_UNTIL_MULTI_TURN:
            model_response_type = GenerativeMultiturnResponse
        elif request_type == RequestType.GREEDY_UNTIL:
            model_response_type = GenerativeResponse
        else:
            raise ValueError(
                f"Loading responses from details for request type {request_type} is currently not supported"
            )

        return model_response_type

    def _run_model(self):
        # Running all requests depending on the model call type (log likelihood, generative, ...)
        # to be able to batch them
        logger.info("--- RUNNING MODEL ---")
        sample_id_to_responses: dict[(SampleUid, MetricCategory), list[ModelResponse]] = collections.defaultdict(list)

        for request_type, requests in self.requests.items():
            logger.info(f"Running {request_type} requests")
            run_model = self.model.get_method_from_request_type(request_type=request_type)
            responses = run_model(requests, override_bs=self.pipeline_parameters.override_batch_size)

            # Storing the responses associated to the same samples together
            for response, request in zip(responses, requests):
                for metric_category in request.metric_categories:
                    sample_id = SampleUid(request.task_name, request.sample_index)
                    sample_id_to_responses[(sample_id, metric_category)].append(response)

        # Cleaning up the model before running metrics
        self.model.cleanup()

        return sample_id_to_responses

    def _get_task(self, task_name: str):
        short_task_name = task_name.rsplit("|", 1)[0]
        return self.task_dict[short_task_name]

    def _compute_metrics(self, sample_id_to_responses):
        # To compute the metrics we first group the samples and task and then by metrics.
        # This way we can batch the metrics computation for each task and metric category

        # This variable will hold the samples grouped by task and metric category
        # example:
        # task_metric_category_groups = {
        #     "task_name": {
        #         "metric_category": {
        #             "ids": [sample_id1, sample_id2, ...],
        #             "responses": [[response1_1, response1_2, ...], [response2_1, response2_2, ...], ...],
        #             "docs": [doc1, doc2, ...]
        #         }
        logger.info("--- COMPUTING METRICS ---")
        task_metric_category_groups = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list))
        )

        for (sample_id, metric_category), sample_responses in sample_id_to_responses.items():
            task_metric_category_groups[sample_id.task_name][metric_category]["ids"].append(sample_id.doc_id_seed)
            task_metric_category_groups[sample_id.task_name][metric_category]["responses"].append(sample_responses)
            task_metric_category_groups[sample_id.task_name][metric_category]["docs"].append(self.docs[sample_id])

        for task_name, samples_per_metric in task_metric_category_groups.items():
            task: LightevalTask = self._get_task(task_name)

            for metric_category, samples in samples_per_metric.items():
                sample_ids = samples["ids"]
                responses = samples["responses"]
                docs = samples["docs"]
                metric_function = task.get_metric_method_from_category(metric_category=metric_category)
                metric_category_metrics = [metric for metric in task.metrics if metric.category == metric_category]

                outputs = metric_function(
                    sample_ids=sample_ids,
                    responses=responses,
                    formatted_docs=docs,
                    metrics=metric_category_metrics,
                )

                for output, doc, response in zip(outputs, docs, responses):
                    self.evaluation_tracker.metrics_logger.log(task_name, output)
                    self.evaluation_tracker.details_logger.log(task_name, task, doc, response, output)

    def save_and_push_results(self):
        logger.info("--- SAVING AND PUSHING RESULTS ---")
        if self.is_main_process():
            self.evaluation_tracker.save()

    def _init_final_dict(self):
        if self.is_main_process():
            if self.final_dict is None:
                self.final_dict = self.evaluation_tracker.generate_final_dict()

    def show_results(self):
        logger.info("--- DISPLAYING RESULTS ---")
        self._init_final_dict()
        if self.is_main_process():
            print(make_results_table(self.final_dict))

    def get_results(self):
        self._init_final_dict()
        return self.final_dict
