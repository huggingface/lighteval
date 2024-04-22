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

from accelerate.commands.launch import launch_command, launch_command_parser  # noqa: I001

from lighteval.commands.parsers import parser_accelerate, parser_nanotron
from lighteval.commands.utils import list_tasks_command


def main():
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # create the parser for the "accelerate" command
    parser_a = subparsers.add_parser("accelerate", help="use accelerate and transformers as backend for evaluation.")
    parser_accelerate(parser_a)

    # create the parser for the "nanotron" command
    parser_b = subparsers.add_parser("nanotron", help="use nanotron as backend for evaluation.")
    parser_nanotron(parser_b)

    subparsers.add_parser("list-tasks", help="List available tasks")

    args = parser.parse_args()

    if args.subcommand == "accelerate":
        if args.num_processes > 1:
            accelerate_args = [
                "--multi_gpu",
                "--num_processes",
                str(args.num_processes),
                "-m",
                "lighteval.main_accelerate",
            ]
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
