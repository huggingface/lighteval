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

""" Example run command:
accelerate config
accelerate launch run_evals_accelerate.py --tasks="leaderboard|hellaswag|5|1" --output_dir "/scratch/evals" --model_args "pretrained=gpt2"
"""
import argparse

from lighteval.main_accelerate import CACHE_DIR, main


def get_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    task_type_group = parser.add_mutually_exclusive_group(required=True)

    # Model type 1) Base model
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

    task_type_group.add_argument("--model_args")
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
    parser.add_argument("--system_prompt", type=str, default=None)
    # Model type 2) TGI
    task_type_group.add_argument("--inference_server_address", type=str)
    parser.add_argument("--inference_server_auth", type=str, default=None)
    # Model type 3) Inference endpoints
    task_type_group.add_argument("--endpoint_model_name", type=str)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--vendor", type=str, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--instance_size", type=str, default=None)
    parser.add_argument("--instance_type", type=str, default=None)
    parser.add_argument("--reuse_existing", default=False, action="store_true")
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
        "--custom_tasks",
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


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()
    main(args)
