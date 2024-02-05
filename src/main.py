import argparse
import os
import random
import shutil
from contextlib import nullcontext
from datetime import timedelta

import numpy as np

from lighteval.evaluator import evaluate, make_results_table
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import hlog, hlog_warn, htrack, htrack_block
from lighteval.models.model_config import EnvConfig, create_model_config
from lighteval.models.model_loader import load_model
from lighteval.tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from lighteval.tasks.registry import Registry, taskinfo_selector
from lighteval.utils import is_accelerate_available, is_tgi_available
from lighteval.utils_parallelism import test_all_gather


if not is_accelerate_available() and not is_tgi_available():
    hlog_warn("Using either accelerate or text-generation to run this script is advised.")

TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("HF_HOME")

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None


def get_parser():
    parser = argparse.ArgumentParser()
    # Model type 1) Base model
    group = parser.add_mutually_exclusive_group(required=True)
    weight_type_group = parser.add_mutually_exclusive_group()

    weight_type_group.add_argument(
        "--delta_weights",
        action="store_true",
        default=False,
        help="set to True of your model should be merged with a base model, also need to provide the base model name",
    )
    weight_type_group.add_argument(
        "--adapter_weights",
        action="store_true",
        default=False,
        help="set to True of your model has been trained with peft, also need to provide the base model name",
    )
    parser.add_argument(
        "--base_model", type=str, default=None, help="name of the base model to be used for delta or adapter weights"
    )

    parser.add_argument("--model_args", required=True)
    parser.add_argument("--model_dtype", type=str, default=None)
    parser.add_argument(
        "--multichoice_continuations_start_space",
        action="store_true",
        help="Whether to force multiple choice continuations to start with a space",
    )
    parser.add_argument(
        "--no_multichoice_continuations_start_space",
        action="store_true",
        help="Whether to force multiple choice continuations to not start with a space",
    )
    parser.add_argument("--use_chat_template", default=False, action="store_true")
    # Model type 2) TGI
    parser.add_argument("--inference_server_address", type=str, default=None)
    parser.add_argument("--inference_server_auth", type=str, default=None)
    # Model type 3) Inference endpoints
    parser.add_argument("--endpoint_model_name", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--vendor", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--instance_size", type=str, default=None)
    parser.add_argument("--instance_type", type=str, default=None)
    # Debug
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--job_id", type=str, help="Optional Job ID for future reference", default="")
    # Saving
    parser.add_argument("--push_results_to_hub", default=False, action="store_true")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--push_details_to_hub", default=False, action="store_true")
    parser.add_argument(
        "--public_run", default=False, action="store_true", help="Push results and details to a public repo"
    )
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument(
        "--results_org",
        type=str,
        help="Hub organisation where you want to store the results. Your current token must have write access to it",
    )
    # Common parameters
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--override_batch_size", type=int, default=-1)
    parser.add_argument("--dataset_loading_processes", type=int, default=1)
    parser.add_argument(
        "--custom_tasks_file",
        type=str,
        default=None,
        help="Path to a file with custom tasks (a TASK list of dict and potentially prompt formating functions)",
    )
    group.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Id of a task, e.g. 'original|mmlu:abstract_algebra|5' or path to a texte file with a list of tasks",
    )
    parser.add_argument("--num_fewshot_seeds", type=int, default=1, help="Number of trials the few shots")
    return parser


@htrack()
def main(args):
    env_config = EnvConfig(token=TOKEN, cache_dir=args.cache_dir)
    evaluation_tracker = EvaluationTracker(hub_results_org=args.results_org, token=TOKEN)
    evaluation_tracker.general_config_logger.log_args_info(
        args.num_fewshot_seeds, args.override_batch_size, args.max_samples, args.job_id
    )

    if args.max_samples:
        hlog(
            "WARNING: --max_samples WAS SET. THESE NUMBERS ARE ONLY PARTIAL AND SHOULD NOT BE USED FOR COMPARISON UNLESS YOU KNOW WHAT YOU ARE DOING."
        )

    with htrack_block("Test all gather"):
        test_all_gather(accelerator)

    with htrack_block("Creating model configuration"):
        model_config = create_model_config(args=args, accelerator=accelerator)

    with htrack_block("Model loading"):
        # We need to load the model in the main process first to avoid downloading the model multiple times
        with accelerator.main_process_first() if accelerator is not None else nullcontext():
            model, model_info = load_model(config=model_config, env_config=env_config)
            evaluation_tracker.general_config_logger.log_model_info(model_info)

    with htrack_block("Tasks loading"):
        with accelerator.main_process_first() if accelerator is not None else nullcontext():
            task_names_list, few_shots_dict = taskinfo_selector(args.tasks)
            task_dict = Registry(cache_dir=env_config.cache_dir).get_task_dict(
                task_names_list, custom_tasks_file=args.custom_tasks_file
            )
            # Loading all the dataset in a distributed manner
            LightevalTask.load_datasets(task_dict.values(), args.dataset_loading_processes)

            evaluation_tracker.task_config_logger.log(task_dict)

            hlog("Loading documents, and requests")
            requests, docs = create_requests_from_tasks(
                task_dict,
                few_shots_dict,
                args.num_fewshot_seeds,
                model,
                args.max_samples,
                evaluation_tracker,
                args.use_chat_template,
            )

    with htrack_block("Setting seeds and waiting for all processes"):
        hlog(f"setting seed to {1234} for random and numpy")
        random.seed(1234)
        np.random.seed(1234)
        if accelerator is not None:
            accelerator.wait_for_everyone()

    with htrack_block("Evaluation"):
        hlog(f"Evaluate on {len(task_names_list)} tasks.")
        evaluation_tracker = evaluate(
            lm=model,
            requests_dict=requests,
            docs=docs,
            task_dict=task_dict,
            override_bs=args.override_batch_size,
            evaluation_tracker=evaluation_tracker,
        )

    if accelerator.is_main_process if accelerator is not None else nullcontext():
        with htrack_block("Compiling and saving results"):
            evaluation_tracker.general_config_logger.log_end_time()
            evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict, bootstrap_iters=1000)
            evaluation_tracker.details_logger.aggregate()

            if args.output_dir:
                evaluation_tracker.save(
                    args.output_dir, args.push_results_to_hub, args.push_details_to_hub, args.public_run
                )

            final_dict = evaluation_tracker.generate_final_dict()

        with htrack_block("Cleaninp up"):
            if args.delta_weights:
                tmp_weights_dir = f"{evaluation_tracker.general_config_logger.model_name}-delta-applied"
                hlog(f"Removing {tmp_weights_dir}")
                shutil.rmtree(tmp_weights_dir)
            if args.adapter_weights:
                tmp_weights_dir = f"{evaluation_tracker.general_config_logger.model_name}-adapter-applied"
                hlog(f"Removing {tmp_weights_dir}")
                shutil.rmtree(tmp_weights_dir)

        print(make_results_table(final_dict))

        return final_dict


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()
    main(args)
