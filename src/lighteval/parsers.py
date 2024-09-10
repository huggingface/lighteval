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


def parser_accelerate(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="CLI tool for lighteval, a lightweight framework for LLM evaluation"
        )

    group = parser.add_mutually_exclusive_group(required=True)
    task_type_group = parser.add_mutually_exclusive_group(required=True)

    # Model type: either use a config file or simply the model name
    task_type_group.add_argument(
        "--model_config_path",
        type=str,
        help="Path to the model config file, e.g. 'examples/model_configs/base_model.yaml'",
    )
    task_type_group.add_argument(
        "--model_args",
        type=str,
        help="Model arguments to pass to the model class, e.g. 'pretrained=gpt2,dtype=float16'",
    )

    # Debug
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate on")
    parser.add_argument("--override_batch_size", type=int, default=-1)
    parser.add_argument("--job_id", type=str, help="Optional Job ID for future reference", default="")

    # Saving
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save the results, fsspec compliant (e.g. s3://bucket/path)",
    )
    parser.add_argument("--save_details", action="store_true", help="Save the details of the run in the output_dir")
    parser.add_argument("--push_to_hub", default=False, action="store_true", help="Set to push the details to the hub")
    parser.add_argument("--push_to_tensorboard", default=False, action="store_true")
    parser.add_argument(
        "--public_run", default=False, action="store_true", help="Push results and details to a public repo"
    )
    parser.add_argument(
        "--results_org",
        type=str,
        help="Hub organisation where you want to store the results. Your current token must have write access to it",
        default=None,
    )
    # Common parameters
    parser.add_argument(
        "--use_chat_template",
        default=False,
        action="store_true",
        help="Use the chat template (from the model's tokenizer) for the prompt",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None, help="System prompt to use, e.g. 'You are a helpful assistant.'"
    )
    parser.add_argument(
        "--dataset_loading_processes", type=int, default=1, help="Number of processes to use for loading the datasets"
    )
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
    parser.add_argument(
        "--cache_dir", type=str, default=CACHE_DIR, help="Cache directory used to store datasets and models"
    )
    parser.add_argument("--num_fewshot_seeds", type=int, default=1, help="Number of trials the few shots")
    return parser


def parser_nanotron(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="CLI tool for lighteval, a lightweight framework for LLM evaluation"
        )

    parser.add_argument(
        "--checkpoint_config_path",
        type=str,
        required=True,
        help="Path to the nanotron checkpoint YAML or python config file, potentially on S3",
    )
    parser.add_argument(
        "--lighteval_config_path",
        type=str,
        help="Path to a YAML or python lighteval config to be used for the evaluation. Lighteval key in nanotron config is ignored!",
        required=True,
    )
    parser.add_argument(
        "--cache_dir", type=str, default=CACHE_DIR, help="Cache directory used to store datasets and models"
    )


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
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to display")
    parser.add_argument("--show_config", default=False, action="store_true", help="Will display the full task config")
    parser.add_argument(
        "--cache_dir", type=str, default=CACHE_DIR, help="Cache directory used to store datasets and models"
    )
