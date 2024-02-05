# flake8: noqa: C901
import argparse
import os
import random

import numpy as np
import torch
from brrr.config import BrrrConfig
from brrr.s3_checkpoints import fs_copy
from brrr.utils import check_env
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import get_config_from_file
from nanotron.logging import get_logger, log_rank
from nanotron.parallel.context import ParallelContext
from nanotron.utils import local_ranks_zero_first

from lighteval.evaluator import evaluate, make_results_table
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import hlog, htrack, htrack_block
from lighteval.models.brrr_models import BRRRModel
from lighteval.models.model_loader import ModelInfo
from lighteval.tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from lighteval.tasks.registry import Registry, get_custom_tasks, taskinfo_selector


logger = get_logger(__name__)

TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("HF_HOME", "/scratch")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-config-path",
        type=str,
        required=True,
        help="Path to the brr checkpoint YAML or python config file, potentially on S3",
    )
    parser.add_argument(
        "--lighteval-override",
        type=str,
        help="Path to an optional YAML or python Lighteval config to override part of the checkpoint Lighteval config",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Local or hub path of an optional tokenizer (if not indicated in the checkpoint)",
    )
    parser.add_argument(
        "--s5cmd-path",
        type=str,
        default="/admin/home/thomwolf/miniconda3/envs/b4r/bin/s5cmd",
        help="Path to s5cmd install",
    )
    parser.add_argument(
        "--s5cmd-numworkers",
        type=int,
        default=64,
        help="s5cmd num workers (optional)",
    )
    parser.add_argument(
        "--s5cmd-concurrency",
        type=int,
        default=10,
        help="s5cmd concurrency (optional)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Cache directory",
    )

    return parser


@htrack()
def main(args):
    cache_dir = args.cache_dir or CACHE_DIR
    check_env()

    dist.initialize_torch_distributed()

    with htrack_block("get config"):
        if not args.checkpoint_config_path.endswith(".yaml"):
            raise ValueError("The checkpoint path should point to a YAML file")
        local_config_path = args.checkpoint_config_path
        if args.checkpoint_config_path.startswith("s3:/"):
            local_config_path = args.checkpoint_config_path.replace("s3:/", cache_dir)
            with local_ranks_zero_first():
                if os.environ.get("LOCAL_RANK", None) == "0":
                    os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
                    fs_copy(args.checkpoint_config_path, local_config_path)

        brrr_config: BrrrConfig = get_config_from_file(local_config_path, config_class=BrrrConfig)

        if args.lighteval_override:
            local_override_path = args.lighteval_override.replace("s3:/", cache_dir)
            if args.lighteval_override.startswith("s3:/"):
                local_override_path = args.lighteval_override.replace("s3:/", cache_dir)
                with local_ranks_zero_first():
                    if os.environ.get("LOCAL_RANK", None) == "0":
                        os.makedirs(os.path.dirname(local_override_path), exist_ok=True)
                        fs_copy(args.lighteval_override, local_override_path)
            lighteval_brrr_config: BrrrConfig = get_config_from_file(local_override_path, config_class=BrrrConfig)
            lighteval_config = lighteval_brrr_config.lighteval
            brrr_config.lighteval = lighteval_config
        else:
            local_override_path = ""
            lighteval_config = brrr_config.lighteval

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
            config=brrr_config.as_dict(),
        )

    with htrack_block("Test all gather"):
        hlog("Test gather tensor")
        # Do a first NCCL sync to warmup and try to avoid Timeout after model/data loading
        log_rank(
            f"[TEST] Running NCCL sync for ranks {list(range(parallel_context.world_pg.size()))}",
            logger=logger,
            level=logging.WARNING,
            group=parallel_context.dp_pg,
            rank=0,
        )
        test_tensor = torch.tensor([dist.get_rank(parallel_context.world_pg)], device=torch.device("cuda"))
        test_tensor_list = [torch.zeros_like(test_tensor) for _ in range(parallel_context.world_pg.size())]
        dist.all_gather(test_tensor_list, test_tensor, group=parallel_context.world_pg, async_op=False)
        dist.barrier()
        log_rank(
            f"[TEST] NCCL sync for ranks {[t.item() for t in test_tensor_list]}",
            logger=logger,
            level=logging.WARNING,
            group=parallel_context.dp_pg,
            rank=0,
        )

        del test_tensor_list
        del test_tensor

    with htrack_block("Model loading"):
        # We need to load the model in the main process first to avoid downloading the model multiple times
        model = BRRRModel(
            checkpoint_path=brrr_config.s3_upload.upload_s3_path / str(brrr_config.general.step),
            model_args=brrr_config.model,
            tokenizer=brrr_config.tokenizer,
            parallel_context=parallel_context,
            parallel_config=lighteval_config.parallelism,
            lighteval_config=lighteval_config,
            batch_size=lighteval_config.batch_size,
            cache_dir=os.environ.get("HF_HOME", "/scratch"),
            debug_one_layer_model=False,
            s5cmd_path=args.s5cmd_path,
            s5cmd_numworkers=args.s5cmd_numworkers,
            s5cmd_concurrency=args.s5cmd_concurrency,
        )
        model_info = ModelInfo(model_name=f"{brrr_config.general.run}/{brrr_config.general.step}")
        evaluation_tracker.general_config_logger.log_model_info(model_info)

    with htrack_block("Tasks loading"):
        with local_ranks_zero_first():
            tasks_selection = lighteval_config.tasks.tasks
            if lighteval_config.tasks.custom_tasks_file:
                _, tasks_groups_dict = get_custom_tasks(lighteval_config.tasks.custom_tasks_file)
                if tasks_groups_dict and lighteval_config.tasks.tasks in tasks_groups_dict:
                    tasks_selection = tasks_groups_dict[lighteval_config.tasks.tasks]

            task_names_list, few_shots_dict = taskinfo_selector(tasks_selection)
            task_dict = Registry(cache_dir=cache_dir).get_task_dict(
                task_names_list, custom_tasks_file=lighteval_config.tasks.custom_tasks_file
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
            )

    with htrack_block("Setting seeds and waiting for all processes"):
        hlog(f"setting seed to {1234} for random and numpy")
        random.seed(1234)
        np.random.seed(1234)
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


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()
    main(args)
