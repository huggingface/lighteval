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
import os
import re
from typing import Optional

from typer import Argument, Option
from typing_extensions import Annotated


TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


def vllm(
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
        str, Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = CACHE_DIR,
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
    job_id: Annotated[
        int, Option(help="Optional job id for future reference.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = 0,
):
    """
    Evaluate models using vllm as backend.
    """
    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    TOKEN = os.getenv("HF_TOKEN")

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
        launcher_type=ParallelismManager.VLLM,
        env_config=env_config,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=-1,  # Cannot override batch size when using vLLM; Configure `max_num_seqs` and `max_num_batched_tokens` in `VLLMModelConfig` instead.
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
    )

    if model_args.endswith(".yaml"):
        with open(model_args, "r") as f:
            config = yaml.safe_load(f)["model"]
        model_args = config["base_params"]["model_args"]
        metric_options = config.get("metric_options", {})
        generation_parameters = GenerationParameters.from_dict(config)
    else:
        generation_parameters = GenerationParameters.from_model_args(model_args)
        # We slice out generation_parameters from model_args to avoid double-counting in the VLLMModelConfig
        model_args = re.sub(r"generation_parameters=\{.*?\},?", "", model_args).strip(",")
        metric_options = {}

    model_args_dict: dict = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in model_args.split(",")}
    model_config = VLLMModelConfig(**model_args_dict, generation_parameters=generation_parameters)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options=metric_options,
    )

    pipeline.evaluate()

    pipeline.show_results()

    results = pipeline.get_results()

    pipeline.save_and_push_results()

    return results
