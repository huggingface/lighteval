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
import asyncio
import collections
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto

import numpy as np
from tqdm import tqdm

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics import apply_metric
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_loader import TransformersModel, load_model
from lighteval.models.model_output import (
    ModelResponse,
)
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import SamplingMethod
from lighteval.utils.imports import is_package_available
from lighteval.utils.parallelism import test_all_gather
from lighteval.utils.utils import make_results_table, remove_reasoning_tags


if is_package_available("accelerate"):
    from accelerate import Accelerator, InitProcessGroupKwargs
else:
    from unittest.mock import Mock

    Accelerator = InitProcessGroupKwargs = Mock()

if is_package_available("nanotron"):
    from nanotron import distributed as dist
    from nanotron.parallel.context import ParallelContext

    from lighteval.models.nanotron.nanotron_model import NanotronLightevalModel


import logging


logger = logging.getLogger(__name__)


class ParallelismManager(Enum):
    ACCELERATE = auto()
    NANOTRON = auto()
    TGI = auto()
    OPENAI = auto()
    VLLM = auto()
    CUSTOM = auto()
    NONE = auto()
    SGLANG = auto()


@dataclass
class PipelineParameters:
    launcher_type: ParallelismManager
    # Env parameters
    job_id: int = 0
    dataset_loading_processes: int = 1
    nanotron_checkpoint_path: str | None = None  # only for nanotron models
    # Dataset
    custom_tasks_directory: str | None = None
    num_fewshot_seeds: int = 1
    max_samples: int | None = None
    cot_prompt: str | None = None
    remove_reasoning_tags: bool = True
    reasoning_tags: str | list[tuple[str, str]] = "[('<think>', '</think>')]"
    load_responses_from_details_date_id: str | None = None
    bootstrap_iters: int = 1000

    def __post_init__(self):  # noqa C901
        if not isinstance(self.reasoning_tags, list):
            try:
                self.reasoning_tags = ast.literal_eval(self.reasoning_tags)
            except ValueError as e:
                raise ValueError(
                    "reasoning_tags must be a list of pair tuples, e.g. [('start_tag', 'end_tag'), ...]. "
                    f"Got {self.reasoning_tags} instead, which caused parsing error {e}."
                )

        # Make sure format is correct
        if not all(isinstance(tag, tuple) and len(tag) == 2 for tag in self.reasoning_tags):
            raise ValueError(
                "reasoning_tags must be a list of pair tuples, e.g. [('start_tag', 'end_tag'), ...]. "
                f"Got {self.reasoning_tags} instead."
            )


class Pipeline:
    def __init__(
        self,
        tasks: str,
        pipeline_parameters: PipelineParameters,
        evaluation_tracker: EvaluationTracker,
        model_config: ModelConfig | None = None,
        model=None,
        metric_options=None,
    ):
        if not (model or model_config):
            raise ValueError("Must provide either a model or model config when creating a pipeline.")

        self.pipeline_parameters = pipeline_parameters
        if self.pipeline_parameters.max_samples:
            logger.warning(
                "--max_samples WAS SET. THESE NUMBERS ARE ONLY PARTIAL AND SHOULD NOT BE USED FOR COMPARISON UNLESS YOU KNOW WHAT YOU ARE DOING."
            )

        self.launcher_type = self.pipeline_parameters.launcher_type
        self._metric_options = metric_options or {}
        self.evaluation_tracker = evaluation_tracker

        # We init tasks first to fail fast if one is badly defined
        self._init_random_seeds()
        self._init_tasks_and_requests(tasks=tasks)

        self.model_config = model_config
        self.accelerator, self.parallel_context = self._init_parallelism_manager()
        self.model = self._init_model(model_config, model)
        # Must occur after model and task init
        self.model._cache._init_registry(self.registry)
        # Must occur after model init
        self._init_accelerator_seeds()

        self.evaluation_tracker.general_config_logger.log_model_info(model_config=self.model.config)

        # Final results
        self.final_dict: dict | None = None

    def _init_parallelism_manager(self):
        accelerator, parallel_context = None, None
        if self.launcher_type == ParallelismManager.ACCELERATE:
            if not is_package_available("accelerate"):
                raise ValueError("You are trying to launch an accelerate model, but accelerate is not installed")
            accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
            test_all_gather(accelerator=accelerator)
        elif self.launcher_type == ParallelismManager.NANOTRON:
            if not is_package_available("nanotron"):
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

        if model is not None and model_config is not None:
            if isinstance(model, LightevalModel):
                raise ValueError(
                    "You are trying to provide both a LightevalModel and a model config. Please provide only one of them."
                )
            return TransformersModel.from_model(
                model=model,
                config=model_config,
                accelerator=self.accelerator,
            )

        elif model is not None:
            if isinstance(model, LightevalModel):
                return model
            raise ValueError("If not providing a model_config, you need to provide a Lighteval model.")

        elif model_config is not None:
            if self.parallel_context:
                return NanotronLightevalModel(
                    checkpoint_path=os.path.dirname(self.pipeline_parameters.nanotron_checkpoint_path)
                    if self.pipeline_parameters.nanotron_checkpoint_path
                    else "",
                    nanotron_config=model_config,
                    parallel_context=self.parallel_context,
                    debug_one_layer_model=False,
                    model_class=None,
                )
            else:
                return load_model(config=model_config)

    def _init_tasks_and_requests(self, tasks: str):
        logger.info("--- LOADING TASKS ---")

        # The registry contains all the potential tasks
        self.registry = Registry(tasks=tasks, custom_tasks=self.pipeline_parameters.custom_tasks_directory)

        # load the tasks from the configs and their datasets
        self.tasks_dict: dict[str, LightevalTask] = self.registry.load_tasks()
        LightevalTask.load_datasets(self.tasks_dict, self.pipeline_parameters.dataset_loading_processes)
        self.documents_dict = {
            task.full_name: task.get_docs(self.pipeline_parameters.max_samples) for _, task in self.tasks_dict.items()
        }

        self.sampling_docs = collections.defaultdict(list)
        for _, docs in self.documents_dict.items():
            for doc in docs:
                for sampling in doc.sampling_methods:
                    self.sampling_docs[sampling].append(doc)

        # If there are metric_options defined from the yaml file,
        # review if they have to be updated.
        if self._metric_options:
            self._update_num_samples(list(self.tasks_dict.values()))

        self.evaluation_tracker.task_config_logger.log(self.tasks_dict)

    def _update_num_samples(self, tasks: list[LightevalTask]):
        """Helper function to update the num_samples of a given metric via the yaml file.
        As it has to be done at the metric level, it's better to update the value per metric.
        It will add a num_samples to the already defined metrics' num_samples if defined in the yaml file.
        As later when constructing the requests the max is taken over the num_samples, this is valid.
        """
        for task in tasks:
            for metric in task.metrics:
                if metric_data := self._metric_options.get(metric.metric_name, None):
                    num_samples = metric_data.get("num_samples", None)
                    if num_samples:
                        task.num_samples = [num_samples]

    def _init_random_seeds(self):
        logger.info("--- INIT SEEDS ---")
        random.seed(1234)
        np.random.seed(1234)

    def _init_accelerator_seeds(self):
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
            max_samples=self.pipeline_parameters.max_samples,
            job_id=str(self.pipeline_parameters.job_id),
        )

        if self.pipeline_parameters.load_responses_from_details_date_id:
            try:
                outputs = self._load_responses_from_details()
            except FileNotFoundError as e:
                logger.warning(
                    f"No responses found for {self.pipeline_parameters.load_responses_from_details_date_id} in details directory: {e}. Running model instead."
                )
                outputs = self._run_model()
        else:
            outputs = self._run_model()

        if self.is_main_process():
            self._post_process_outputs(outputs)
            self._compute_metrics(outputs)

            self.evaluation_tracker.general_config_logger.log_end_time()
            self.evaluation_tracker.metrics_logger.aggregate(
                task_dict=self.tasks_dict, bootstrap_iters=self.pipeline_parameters.bootstrap_iters
            )
            self.evaluation_tracker.details_logger.aggregate()

    async def _run_model_async(self):
        outputs = {}
        for sampling_method, docs in self.sampling_docs.items():
            logger.info(f"Running {sampling_method} requests")
            match sampling_method:
                case SamplingMethod.GENERATIVE:
                    model_outputs = await self.model.greedy_until(docs)
                    outputs[sampling_method] = model_outputs
                case SamplingMethod.LOGPROBS:
                    model_outputs = await self.model.loglikelihood(docs)
                    outputs[sampling_method] = model_outputs

        return outputs

    def _run_model_sync(self):
        # Running all requests depending on the model call type (log likelihood, generative, ...)
        # to be able to batch them
        outputs = {}
        for sampling_method, docs in self.sampling_docs.items():
            logger.info(f"Running {sampling_method} requests")
            match sampling_method:
                case SamplingMethod.GENERATIVE:
                    model_outputs = self.model.greedy_until(docs)
                    outputs[sampling_method] = model_outputs
                case SamplingMethod.LOGPROBS:
                    model_outputs = self.model.loglikelihood(docs)
                    outputs[sampling_method] = model_outputs
                case SamplingMethod.PERPLEXITY:
                    model_outputs = self.model.loglikelihood_rolling(docs)
                    outputs[sampling_method] = model_outputs

        return outputs

    def _run_model(self):
        # Running all requests depending on the model call type (log likelihood, generative, ...)
        # to be able to batch them
        logger.info("--- RUNNING MODEL ---")

        if self.model.is_async:
            outputs = asyncio.run(self._run_model_async())
        else:
            outputs = self._run_model_sync()

        # Cleaning up the model before running metrics
        self.model.cleanup()

        return outputs

    def _post_process_outputs(self, sampling_method_responses: dict[str, list[ModelResponse]]):
        # Removes reasoning tags if needed
        logger.info("--- POST-PROCESSING MODEL RESPONSES ---")

        if self.pipeline_parameters.remove_reasoning_tags:
            for _, responses in sampling_method_responses.items():
                for response in responses:
                    response.text_post_processed = [
                        remove_reasoning_tags(
                            text=text,
                            tag_pairs=self.pipeline_parameters.reasoning_tags,
                        )
                        for text in response.text
                    ]

    def _compute_metrics(self, sampling_method_responses: dict[str, list[ModelResponse]]):
        # To compute the metrics we first group the samples and task and then by metrics.
        # This way we can batch the metrics computation for each task and metric category

        # This variable will hold the samples grouped by task and metric category
        # example:
        # task_metric_category_groups = {
        #     "gsm8k_1": {
        #         "GENERATIVE": [
        #             (doc1, response1), (doc2, response2), ...,
        #         }
        #         "LOGLIKELIHOOD": [
        #             (doc1, response1), (doc2, response2), ...,
        #         ]
        logger.info("--- COMPUTING METRICS ---")
        task_metric_category_groups = collections.defaultdict(lambda: collections.defaultdict(list))

        for sampling_method, model_responses in sampling_method_responses.items():
            for doc, model_reponse in zip(self.sampling_docs[sampling_method], model_responses):
                task_metric_category_groups[doc.task_name][sampling_method].append((doc, model_reponse))

        for task_name, samples_per_method in task_metric_category_groups.items():
            task: LightevalTask = self.tasks_dict[task_name]
            for sampling_method, samples in samples_per_method.items():
                metric_category_metrics = [metric for metric in task.metrics if metric.category == sampling_method]

                docs = [doc for doc, _ in samples]
                responses = [response for _, response in samples]

                outputs = apply_metric(
                    docs=docs,
                    responses=responses,
                    metrics=metric_category_metrics,
                )

                for output, doc, response in zip(outputs, docs, responses):
                    self.evaluation_tracker.metrics_logger.log(task_name, output)
                    self.evaluation_tracker.details_logger.log(task_name, doc, response, output)

    def _load_responses_from_details(self):
        logger.info("--- LOADING RESPONSES FROM DETAILS ---")
        model_responses = {}
        tasks_names = list(self.tasks_dict.keys())
        sampling_methods = list(self.sampling_docs.keys())

        if len(sampling_methods) > 1:
            raise ValueError(
                "Loading responses from details when there are multiple request types is currently not supported"
            )

        assert self.pipeline_parameters.load_responses_from_details_date_id is not None

        details_datasets = self.evaluation_tracker.load_details_datasets(
            self.pipeline_parameters.load_responses_from_details_date_id, tasks_names
        )

        for _, dataset in tqdm(details_datasets.items(), desc="Loading responses from details for tasks"):
            for sampling_method in sampling_methods:
                model_responses[sampling_method] = [
                    ModelResponse(**model_response["model_response"]) for model_response in dataset
                ]

        return model_responses

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

    def get_details(self):
        return self.evaluation_tracker.details_logger.details
