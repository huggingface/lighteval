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
from dataclasses import asdict
from pprint import pformat

from lighteval.parsers import parser_accelerate, parser_nanotron, parser_utils_tasks
from lighteval.tasks.registry import Registry, taskinfo_selector


CACHE_DIR = os.getenv("HF_HOME")


def cli_evaluate():
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # Subparser for the "accelerate" command
    parser_a = subparsers.add_parser("accelerate", help="use accelerate and transformers as backend for evaluation.")
    parser_accelerate(parser_a)

    # Subparser for the "nanotron" command
    parser_b = subparsers.add_parser("nanotron", help="use nanotron as backend for evaluation.")
    parser_nanotron(parser_b)

    # Subparser for task utils functions
    parser_c = subparsers.add_parser("tasks", help="display information about available tasks and samples.")
    parser_utils_tasks(parser_c)

    args = parser.parse_args()

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as main_accelerate

        main_accelerate(args)

    elif args.subcommand == "nanotron":
        from lighteval.main_nanotron import main as main_nanotron

        main_nanotron(args.checkpoint_config_path, args.lighteval_config_path, args.cache_dir)

    elif args.subcommand == "tasks":
        if args.list:
            Registry(cache_dir="").print_all_tasks()

        if args.inspect:
            print(f"Loading the tasks dataset to cache folder: {args.cache_dir}")
            print(
                "All examples will be displayed without few shot, as few shot sample construction requires loading a model and using its tokenizer. "
            )
            # Loading task
            task_names_list, _ = taskinfo_selector(args.inspect)
            task_dict = Registry(cache_dir=args.cache_dir).get_task_dict(task_names_list)
            for name, task in task_dict.items():
                print("-" * 10, name, "-" * 10)
                if args.show_config:
                    print("-" * 10, "CONFIG")
                    task.cfg.print()
                for ix, sample in enumerate(task.eval_docs()[: int(args.num_samples)]):
                    if ix == 0:
                        print("-" * 10, "SAMPLES")
                    print(f"-- sample {ix} --")
                    print(pformat(asdict(sample), indent=1))

    else:
        print("You did not provide any argument. Exiting")


if __name__ == "__main__":
    cli_evaluate()
