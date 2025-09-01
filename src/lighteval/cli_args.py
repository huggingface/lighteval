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

"""
Common CLI argument types for LightEval main files.
This module exports pre-defined argument types to reduce redundancy across main_*.py files.
"""

from dataclasses import dataclass
from typing import Any, Optional

from typer import Argument, Option
from typing_extensions import Annotated


# Help panel names for consistent organization
HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


@dataclass
class Arg:
    """Base class for CLI arguments with type and default value."""

    type: Annotated
    default: Any


# Common Parameters (HELP_PANEL_NAME_1)
dataset_loading_processes = Arg(
    type=Annotated[
        int,
        Option(
            help="Number of parallel processes to use for loading datasets. Higher values can speed up dataset loading but use more memory.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ],
    default=1,
)

custom_tasks = Arg(
    type=Annotated[
        Optional[str],
        Option(
            help="Path to a Python file containing custom task definitions. The file should define a TASKS_TABLE with LightevalTaskConfig objects.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ],
    default=None,
)

num_fewshot_seeds = Arg(
    type=Annotated[
        int,
        Option(
            help="Number of different random seeds to use for few-shot evaluation. Each seed will generate different few-shot examples, providing more robust evaluation.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ],
    default=1,
)

load_responses_from_details_date_id = Arg(
    type=Annotated[
        Optional[str],
        Option(
            help="Load previously generated model responses from a specific evaluation run instead of running the model. Use the timestamp/date_id from a previous run's details directory.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ],
    default=None,
)

remove_reasoning_tags = Arg(
    type=Annotated[
        bool,
        Option(
            help="Whether to remove reasoning tags from model responses before computing metrics.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ],
    default=True,
)

reasoning_tags = Arg(
    type=Annotated[
        str,
        Option(
            help="List of reasoning tag pairs to remove from responses, formatted as a Python list of tuples.",
            rich_help_panel=HELP_PANEL_NAME_1,
        ),
    ],
    default="[('<think>', '</think>')]",
)


# Logging Parameters (HELP_PANEL_NAME_2)
output_dir = Arg(
    type=Annotated[
        str,
        Option(
            help="Directory where evaluation results and details will be saved. Supports fsspec-compliant paths (local, s3, hf hub, etc.).",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default="results",
)

results_path_template = Arg(
    type=Annotated[
        str | None,
        Option(
            help="Custom template for results file path. Available variables: {output_dir}, {org}, {model}. Example: '{output_dir}/experiments/{org}_{model}' creates results in a subdirectory.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=None,
)

push_to_hub = Arg(
    type=Annotated[
        bool,
        Option(
            help="Whether to push evaluation results and details to the Hugging Face Hub. Requires --results-org to be set.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=False,
)

push_to_tensorboard = Arg(
    type=Annotated[
        bool,
        Option(
            help="Whether to create and push TensorBoard logs to the Hugging Face Hub. Requires --results-org to be set.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=False,
)

public_run = Arg(
    type=Annotated[
        bool,
        Option(
            help="Whether to make the uploaded results and details public on the Hugging Face Hub. If False, datasets will be private.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=False,
)

results_org = Arg(
    type=Annotated[
        Optional[str],
        Option(
            help="Hugging Face organization where results will be pushed. Required when using --push-to-hub or --push-to-tensorboard.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=None,
)

save_details = Arg(
    type=Annotated[
        bool,
        Option(
            help="Whether to save detailed per-sample results including model inputs, outputs, and metrics. Useful for analysis and debugging.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=False,
)

wandb = Arg(
    type=Annotated[
        bool,
        Option(
            help="Whether to log results to Weights & Biases (wandb) or Trackio. Configure with environment variables: WANDB_PROJECT, WANDB_SPACE_ID, etc. See wandb docs for full configuration options.",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ],
    default=False,
)


# Debug Parameters (HELP_PANEL_NAME_3)
max_samples = Arg(
    type=Annotated[
        Optional[int],
        Option(
            help="Maximum number of samples to evaluate per task. Useful for quick testing or debugging. If None, evaluates on all available samples.",
            rich_help_panel=HELP_PANEL_NAME_3,
        ),
    ],
    default=None,
)

job_id = Arg(
    type=Annotated[
        int,
        Option(
            help="Optional job identifier for tracking and organizing multiple evaluation runs. Useful in cluster environments.",
            rich_help_panel=HELP_PANEL_NAME_3,
        ),
    ],
    default=0,
)


# Common argument patterns
tasks = Arg(
    type=Annotated[
        str,
        Argument(
            help="Comma-separated list of tasks to evaluate. Format: 'task1,task2' or 'suite|task|version|split'. Use 'lighteval tasks list' to see available tasks."
        ),
    ],
    default=None,  # Required argument, no default
)

model_args = Arg(
    type=Annotated[
        str,
        Argument(
            help="Model configuration in key=value format (e.g., 'pretrained=model_name,device=cuda') or path to YAML config file. See examples/model_configs/ for template files."
        ),
    ],
    default=None,  # Required argument, no default
)
