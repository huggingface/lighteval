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

from typing import Optional

from typer import Argument, Option
from typing_extensions import Annotated


# Help panel names for consistent organization
HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


# Common Parameters (HELP_PANEL_NAME_1)
DatasetLoadingProcesses = Annotated[
    int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANEL_NAME_1)
]

CustomTasks = Annotated[Optional[str], Option(help="Path to custom tasks file.", rich_help_panel=HELP_PANEL_NAME_1)]

NumFewshotSeeds = Annotated[
    int, Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANEL_NAME_1)
]

LoadResponsesFromDetailsDateId = Annotated[
    Optional[str], Option(help="Load responses from details directory.", rich_help_panel=HELP_PANEL_NAME_1)
]

RemoveReasoningTags = Annotated[
    bool,
    Option(
        help="Remove reasoning tags from responses.",
        rich_help_panel=HELP_PANEL_NAME_1,
    ),
]

ReasoningTags = Annotated[
    str,
    Option(
        help="List of reasoning tags (provided as pairs) to remove from responses.",
        rich_help_panel=HELP_PANEL_NAME_1,
    ),
]


# Logging Parameters (HELP_PANEL_NAME_2)
OutputDir = Annotated[str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANEL_NAME_2)]

ResultsPathTemplate = Annotated[
    str | None,
    Option(
        help="Template path for where to save the results, you have access to 3 variables, `output_dir`, `org` and `model`. for example a template can be `'{output_dir}/1234/{org}+{model}'`",
        rich_help_panel=HELP_PANEL_NAME_2,
    ),
]

PushToHub = Annotated[bool, Option(help="Push results to the huggingface hub.", rich_help_panel=HELP_PANEL_NAME_2)]

PushToTensorboard = Annotated[bool, Option(help="Push results to tensorboard.", rich_help_panel=HELP_PANEL_NAME_2)]

PublicRun = Annotated[
    bool, Option(help="Push results and details to a public repo.", rich_help_panel=HELP_PANEL_NAME_2)
]

ResultsOrg = Annotated[
    Optional[str], Option(help="Organization to push results to.", rich_help_panel=HELP_PANEL_NAME_2)
]

SaveDetails = Annotated[
    bool, Option(help="Save detailed, sample per sample, results.", rich_help_panel=HELP_PANEL_NAME_2)
]

Wandb = Annotated[
    bool,
    Option(
        help="Push results to wandb or trackio if available. We use env variable to configure trackio or wandb. see here: https://docs.wandb.ai/guides/track/environment-variables/, https://github.com/gradio-app/trackio",
        rich_help_panel=HELP_PANEL_NAME_2,
    ),
]


# Debug Parameters (HELP_PANEL_NAME_3)
MaxSamples = Annotated[
    Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANEL_NAME_3)
]

JobId = Annotated[int, Option(help="Optional job id for future reference.", rich_help_panel=HELP_PANEL_NAME_3)]


# Common argument patterns
Tasks = Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")]

ModelArgs = Annotated[
    str,
    Argument(
        help="Model arguments in the form key1=value1,key2=value2,... or path to yaml config file (see examples/model_configs/transformers_model.yaml)"
    ),
]


# Default values for common arguments
DEFAULT_VALUES = {
    "dataset_loading_processes": 1,
    "custom_tasks": None,
    "num_fewshot_seeds": 1,
    "load_responses_from_details_date_id": None,
    "remove_reasoning_tags": True,
    "reasoning_tags": "[('<think>', '</think>')]",
    "output_dir": "results",
    "results_path_template": None,
    "push_to_hub": False,
    "push_to_tensorboard": False,
    "public_run": False,
    "results_org": None,
    "save_details": False,
    "wandb": False,
    "max_samples": None,
    "job_id": 0,
}
