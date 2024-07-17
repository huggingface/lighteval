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

from lighteval.parsers import parser_accelerate, parser_nanotron
from lighteval.tasks.registry import Registry


def cli_evaluate():
    parser = argparse.ArgumentParser(description="CLI tool for lighteval, a lightweight framework for LLM evaluation")
    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # create the parser for the "accelerate" command
    parser_a = subparsers.add_parser("accelerate", help="use accelerate and transformers as backend for evaluation.")
    parser_accelerate(parser_a)

    # create the parser for the "nanotron" command
    parser_b = subparsers.add_parser("nanotron", help="use nanotron as backend for evaluation.")
    parser_nanotron(parser_b)

    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")

    args = parser.parse_args()

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as main_accelerate

        main_accelerate(args)
        return

    if args.subcommand == "nanotron":
        from lighteval.main_nanotron import main as main_nanotron

        main_nanotron(args.checkpoint_config_path, args.lighteval_override, args.cache_dir)
        return

    if args.list_tasks:
        Registry(cache_dir="").print_all_tasks()
        return


if __name__ == "__main__":
    cli_evaluate()
