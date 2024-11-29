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
import argparse
import os


TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("HF_HOME")


def parser_utils_tasks(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="CLI tool for lighteval, a lightweight framework for LLM evaluation"
        )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--list", action="store_true", help="List available tasks")
    group.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Id of tasks or path to a text file with a list of tasks (e.g. 'original|mmlu:abstract_algebra|5') for which you want to manually inspect samples.",
    )
    parser.add_argument("--custom_tasks", type=str, default=None, help="Path to a file with custom tasks")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to display")
    parser.add_argument("--show_config", default=False, action="store_true", help="Will display the full task config")
    parser.add_argument(
        "--cache_dir", type=str, default=CACHE_DIR, help="Cache directory used to store datasets and models"
    )
