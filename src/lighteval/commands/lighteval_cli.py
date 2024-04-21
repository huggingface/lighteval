#!/usr/bin/env python

# MIT License

# Copyright (c) 2024 Taratra D. RAHARISON and The HuggingFace Team

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

import argparse
import os
from pprint import pprint

from accelerate.commands.launch import launch_command, launch_command_parser  # noqa: I001

from lighteval.commands.utils import list_tasks_command


TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("HF_HOME")

def parser_accelerate(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    task_type_group = parser.add_mutually_exclusive_group(required=True)

    # Model type: either use a config file or simply the model name
    task_type_group.add_argument("--model_config_path")
    task_type_group.add_argument("--model_args")


    parser.add_argument("--num_processes", type=int, default=1)

    # Debug
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--override_batch_size", type=int, default=-1)
    parser.add_argument("--job_id", type=str, help="Optional Job ID for future reference", default="")
    # Saving
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--use_chat_template", default=False, action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
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


def parser_nanotron(parser):
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
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory",
    )


def main():
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    subparsers = parser.add_subparsers(help='help for subcommand', dest="subcommand")

    # create the parser for the "accelerate" command
    parser_a = subparsers.add_parser('accelerate', help='use accelerate and transformers as backend for evaluation.')
    parser_accelerate(parser_a)

    # create the parser for the "nanotron" command
    parser_b = subparsers.add_parser('nanotron', help='use nanotron as backend for evaluation.')
    parser_nanotron(parser_b)

    parser_c = subparsers.add_parser('list-tasks', help='List available tasks')

    args = parser.parse_args()

    if args.subcommand == "accelerate":
        if args.num_processes > 1:
            accelerate_args = ["--multi_gpu", "--num_processes", str(args.num_processes), "-m", "lighteval.main_accelerate"]
        else:
            accelerate_args = ["--num_processes", "1", "-m", "lighteval.main_accelerate"]

        for key, value in vars(args).items():
            if value is not None and key != "subcommand" and value is not False:
                accelerate_args.extend([f"--{str(key)}", str(value)])

        args_accelerate = launch_command_parser().parse_args(accelerate_args)
        launch_command(args_accelerate)
        return

    if args.subcommand == "nanotron":
        from lighteval.main_nanotron import main as main_nanotron
        main_nanotron(args)
        return

    if args.subcommand == "list-tasks":
        list_tasks_command()
        return

if __name__ == "__main__":
    main()
