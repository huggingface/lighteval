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

# flake8: noqa: C901
import os
import random
from typing import Optional, Type

import numpy as np

from lighteval.evaluator import evaluate, make_results_table
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import hlog, htrack, htrack_block
from lighteval.models.model_config import EnvConfig
from lighteval.models.model_loader import ModelInfo
from lighteval.models.nanotron_model import NanotronLightevalModel
from lighteval.tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from lighteval.tasks.registry import Registry, get_custom_tasks, taskinfo_selector
from lighteval.utils import NO_NANOTRON_ERROR_MSG, is_nanotron_available
from lighteval.utils_parallelism import test_all_gather


if not is_nanotron_available():
    raise ImportError(NO_NANOTRON_ERROR_MSG)

from nanotron import distributed as dist
from nanotron.config import Config, LightEvalConfig, get_config_from_file
from nanotron.logging import get_logger
from nanotron.parallel.context import ParallelContext
from nanotron.utils import local_ranks_zero_first


logger = get_logger(__name__)

SEED = 1234
TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("HF_HOME", "/scratch")


@htrack()
def main(
    checkpoint_config_path: str,
    lighteval_config_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    config_cls: Type = Config,
    model_config_cls: Optional[Type] = None,
    model_cls: Optional[Type] = None,
):
    if cache_dir is None:
        cache_dir = CACHE_DIR

    env_config = EnvConfig(token=TOKEN, cache_dir=cache_dir)

    dist.initialize_torch_distributed()

    with htrack_block("get config"):
        if not checkpoint_config_path.endswith(".yaml"):
            raise ValueError("The checkpoint path should point to a YAML file")

        nanotron_config: config_cls = get_config_from_file(
            checkpoint_config_path,
            config_class=config_cls,
            model_config_class=model_config_cls,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )

        if lighteval_config_path:
            lighteval_config: config_cls = get_config_from_file(lighteval_config_path, config_class=LightEvalConfig)
            nanotron_config.lighteval = lighteval_config
        else:
            lighteval_config = nanotron_config.lighteval

        parallel_context = ParallelContext(
            tensor_parallel_size=lighteval_config.parallelism.tp,
            pipeline_parallel_size=lighteval_config.parallelism.pp,
            data_parallel_size=lighteval_config.parallelism.dp,
        )

        evaluation_tracker = EvaluationTracker(token=TOKEN)
        evaluation_tracker.general_config_logger.log_args_info(
            num_fewshot_seeds=1,
            override_batch_size=None,
            max_samples=lighteval_config.tasks.max_samples,
            job_id=os.environ.get("SLURM_JOB_ID", None),
            config=nanotron_config,
        )

    with htrack_block("Test all gather"):
        test_all_gather(parallel_context=parallel_context)

    with htrack_block("Model loading"):
        # We need to load the model in the main process first to avoid downloading the model multiple times
        model = NanotronLightevalModel(
            checkpoint_path=os.path.dirname(checkpoint_config_path),
            model_args=nanotron_config.model,
            tokenizer=nanotron_config.tokenizer,
            parallel_context=parallel_context,
            parallel_config=lighteval_config.parallelism,
            lighteval_config=lighteval_config,
            batch_size=lighteval_config.batch_size,
            debug_one_layer_model=False,
            model_class=model_cls,
            env_config=env_config,
        )
        model_info = ModelInfo(model_name=f"{nanotron_config.general.run}/{nanotron_config.general.step}")
        evaluation_tracker.general_config_logger.log_model_info(model_info)

    with htrack_block("Tasks loading"):
        with local_ranks_zero_first():
            tasks_selection = lighteval_config.tasks.tasks
            if lighteval_config.tasks.custom_tasks:
                _, tasks_groups_dict = get_custom_tasks(lighteval_config.tasks.custom_tasks)
                if tasks_groups_dict and lighteval_config.tasks.tasks in tasks_groups_dict:
                    tasks_selection = tasks_groups_dict[lighteval_config.tasks.tasks]

            task_names_list, few_shots_dict = taskinfo_selector(tasks_selection)
            task_dict = Registry(cache_dir=cache_dir).get_task_dict(
                task_names_list,
                custom_tasks=lighteval_config.tasks.custom_tasks,
            )
            # Loading all the dataset in a distributed manner
            LightevalTask.load_datasets(task_dict.values(), lighteval_config.tasks.dataset_loading_processes)

            evaluation_tracker.task_config_logger.log(task_dict)

            hlog("Loading documents, and requests")
            requests, docs = create_requests_from_tasks(
                task_dict=task_dict,
                fewshot_dict=few_shots_dict,
                num_fewshot_seeds=lighteval_config.tasks.num_fewshot_seeds or 1,
                lm=model,
                max_samples=lighteval_config.tasks.max_samples,
                evaluation_tracker=evaluation_tracker,
                use_chat_template=False,
                system_prompt=None,
            )

    with htrack_block("Setting seeds and waiting for all processes"):
        hlog(f"setting seed to {SEED} for random and numpy")
        random.seed(SEED)
        np.random.seed(SEED)
        dist.barrier()

    with htrack_block("Evaluation"):
        hlog(f"Evaluate on {len(task_names_list)} tasks.")
        evaluation_tracker = evaluate(
            lm=model,
            requests_dict=requests,
            docs=docs,
            task_dict=task_dict,
            override_bs=lighteval_config.batch_size,
            evaluation_tracker=evaluation_tracker,
        )

    if dist.get_rank(parallel_context.world_pg) == 0:
        with htrack_block("Compiling and saving results"):
            evaluation_tracker.general_config_logger.log_end_time()
            evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict, bootstrap_iters=1000)
            evaluation_tracker.details_logger.aggregate()

            if lighteval_config.logging.local_output_path:
                evaluation_tracker.save(
                    output_dir=lighteval_config.logging.local_output_path,
                    push_results_to_hub=lighteval_config.logging.push_results_to_hub,
                    push_details_to_hub=lighteval_config.logging.push_details_to_hub,
                    public=False,
                    push_results_to_tensorboard=lighteval_config.logging.push_results_to_tensorboard,
                )

            final_dict = evaluation_tracker.generate_final_dict()

        hlog(make_results_table(final_dict))

        return final_dict
