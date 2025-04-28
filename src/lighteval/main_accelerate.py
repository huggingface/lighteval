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

import logging
import os
from typing import Optional

from typer import Argument, Option
from typing_extensions import Annotated


logger = logging.getLogger(__name__)

TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = os.getenv("HF_HOME")

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


def accelerate(  # noqa C901
    # === general ===
    model_args: Annotated[
        str,
        Argument(
            help="Model arguments in the form key1=value1,key2=value2,... or path to yaml config file (see examples/model_configs/transformers_model.yaml)"
        ),
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    use_chat_template: Annotated[
        bool, Option(help="Use chat template for evaluation.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = False,
    system_prompt: Annotated[
        Optional[str], Option(help="Use system prompt for evaluation.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    cache_dir: Annotated[
        Optional[str], Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    num_fewshot_seeds: Annotated[
        int, Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    load_responses_from_details_date_id: Annotated[
        Optional[str], Option(help="Load responses from details directory.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    # === saving ===
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = "results",
    push_to_hub: Annotated[
        bool, Option(help="Push results to the huggingface hub.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    push_to_tensorboard: Annotated[
        bool, Option(help="Push results to tensorboard.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    public_run: Annotated[
        bool, Option(help="Push results and details to a public repo.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    results_org: Annotated[
        Optional[str], Option(help="Organization to push results to.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = None,
    save_details: Annotated[
        bool, Option(help="Save detailed, sample per sample, results.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = False,
    # === debug ===
    max_samples: Annotated[
        Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = None,
    override_batch_size: Annotated[
        int, Option(help="Override batch size for evaluation.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = -1,
    job_id: Annotated[
        int, Option(help="Optional job id for future reference.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = 0,
):
    """
    Evaluate models using accelerate and transformers as backend.
    """
    from datetime import timedelta

    import torch
    import yaml
    from accelerate import Accelerator, InitProcessGroupKwargs

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.transformers.adapter_model import AdapterModelConfig
    from lighteval.models.transformers.delta_model import DeltaModelConfig
    from lighteval.models.transformers.transformers_model import BitsAndBytesConfig, TransformersModelConfig
    from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    cache_dir = CACHE_DIR

    env_config = EnvConfig(token=TOKEN, cache_dir=cache_dir)

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=env_config,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=override_batch_size,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
    )

    # TODO (nathan): better handling of model_args
    if model_args.endswith(".yaml"):
        with open(model_args, "r") as f:
            config = yaml.safe_load(f)["model"]

        # Creating optional quantization configuration
        if config["base_params"]["dtype"] == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif config["base_params"]["dtype"] == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        # We extract the model args
        args_dict = {k.split("=")[0]: k.split("=")[1] for k in config["base_params"]["model_args"].split(",")}

        args_dict["generation_parameters"] = GenerationParameters.from_dict(config)

        # We store the relevant other args
        args_dict["base_model"] = config["merged_weights"]["base_model"]
        args_dict["compile"] = bool(config["base_params"]["compile"])
        args_dict["dtype"] = config["base_params"]["dtype"]
        args_dict["accelerator"] = accelerator
        args_dict["quantization_config"] = quantization_config
        args_dict["batch_size"] = override_batch_size
        args_dict["multichoice_continuations_start_space"] = config["base_params"][
            "multichoice_continuations_start_space"
        ]
        args_dict["use_chat_template"] = use_chat_template

        # Keeping only non null params
        args_dict = {k: v for k, v in args_dict.items() if v is not None}

        if config["merged_weights"].get("delta_weights", False):
            if config["merged_weights"]["base_model"] is None:
                raise ValueError("You need to specify a base model when using delta weights")
            model_config = DeltaModelConfig(**args_dict)
        elif config["merged_weights"].get("adapter_weights", False):
            if config["merged_weights"]["base_model"] is None:
                raise ValueError("You need to specify a base model when using adapter weights")
            model_config = AdapterModelConfig(**args_dict)
        elif config["merged_weights"]["base_model"] not in ["", None]:
            raise ValueError("You can't specify a base model if you are not using delta/adapter weights")
        else:
            model_config = TransformersModelConfig(**args_dict)
    else:
        model_args_dict: dict = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in model_args.split(",")}
        model_args_dict["accelerator"] = accelerator
        model_args_dict["use_chat_template"] = use_chat_template
        model_args_dict["compile"] = bool(model_args_dict["compile"]) if "compile" in model_args_dict else False
        model_config = TransformersModelConfig(**model_args_dict)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    results = pipeline.get_results()

    pipeline.save_and_push_results()

    return results
