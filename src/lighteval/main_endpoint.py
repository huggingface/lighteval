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
from typing import Optional

import typer
from typer import Argument, Option
from typing_extensions import Annotated


app = typer.Typer()


TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")

HELP_PANNEL_NAME_1 = "Common Paramaters"
HELP_PANNEL_NAME_2 = "Logging Parameters"
HELP_PANNEL_NAME_3 = "Debug Paramaters"
HELP_PANNEL_NAME_4 = "Modeling Paramaters"


@app.command(rich_help_panel="Evaluation Backends")
def openai(
    # === general ===
    model_name: Annotated[
        str, Argument(help="The model name to evaluate (has to be available through the openai API.")
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    system_prompt: Annotated[
        Optional[str], Option(help="Use system prompt for evaluation.", rich_help_panel=HELP_PANNEL_NAME_4)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = None,
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = CACHE_DIR,
    num_fewshot_seeds: Annotated[
        int, Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    # === saving ===
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = "results",
    push_to_hub: Annotated[
        bool, Option(help="Push results to the huggingface hub.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    push_to_tensorboard: Annotated[
        bool, Option(help="Push results to tensorboard.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    public_run: Annotated[
        bool, Option(help="Push results and details to a public repo.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    results_org: Annotated[
        Optional[str], Option(help="Organization to push results to.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = None,
    save_details: Annotated[
        bool, Option(help="Save detailed, sample per sample, results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    # === debug ===
    max_samples: Annotated[
        Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = None,
    job_id: Annotated[
        int, Option(help="Optional job id for future refenrence.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = 0,
):
    """
    Evaluate OPENAI models.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_config import OpenAIModelConfig
    from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    env_config = EnvConfig(token=TOKEN, cache_dir=cache_dir)
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )

    parallelism_manager = ParallelismManager.OPENAI
    model_config = OpenAIModelConfig(model=model_name)

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        env_config=env_config,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=-1,  # Cannot override batch size when using OpenAI
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=False,  # Cannot use chat template when using OpenAI
        system_prompt=system_prompt,
    )
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


@app.command(rich_help_panel="Evaluation Backends")
def inference_endpoint(
    # === general ===
    model_config_path: Annotated[
        str, Argument(help="Path to model config yaml file. (examples/model_configs/endpoint_model.yaml)")
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    use_chat_template: Annotated[
        bool, Option(help="Use chat template for evaluation.", rich_help_panel=HELP_PANNEL_NAME_4)
    ] = False,
    system_prompt: Annotated[
        Optional[str], Option(help="Use system prompt for evaluation.", rich_help_panel=HELP_PANNEL_NAME_4)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = None,
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = CACHE_DIR,
    num_fewshot_seeds: Annotated[
        int, Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    # === saving ===
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = "results",
    push_to_hub: Annotated[
        bool, Option(help="Push results to the huggingface hub.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    push_to_tensorboard: Annotated[
        bool, Option(help="Push results to tensorboard.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    public_run: Annotated[
        bool, Option(help="Push results and details to a public repo.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    results_org: Annotated[
        Optional[str], Option(help="Organization to push results to.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = None,
    save_details: Annotated[
        bool, Option(help="Save detailed, sample per sample, results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    # === debug ===
    max_samples: Annotated[
        Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = None,
    override_batch_size: Annotated[
        int, Option(help="Override batch size for evaluation.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = -1,
    job_id: Annotated[
        int, Option(help="Optional job id for future refenrence.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = 0,
):
    """
    Evaluate models using inference-endpoints as backend.
    """
    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_config import (
        InferenceEndpointModelConfig,
    )
    from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    env_config = EnvConfig(token=TOKEN, cache_dir=cache_dir)
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )

    # TODO (nathan): better handling of model_args

    parallelism_manager = ParallelismManager.TGI

    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)["model"]

    # Find a way to add this back
    # if config["base_params"].get("endpoint_name", None):
    #    return InferenceModelConfig(model=config["base_params"]["endpoint_name"])
    all_params = {
        "model_name": config["base_params"].get("model_name", None),
        "endpoint_name": config["base_params"].get("endpoint_name", None),
        "model_dtype": config["base_params"].get("dtype", None),
        "revision": config["base_params"].get("revision", None) or "main",
        "should_reuse_existing": config["base_params"].get("should_reuse_existing"),
        "accelerator": config.get("instance", {}).get("accelerator", None),
        "region": config.get("instance", {}).get("region", None),
        "vendor": config.get("instance", {}).get("vendor", None),
        "instance_size": config.get("instance", {}).get("instance_size", None),
        "instance_type": config.get("instance", {}).get("instance_type", None),
        "namespace": config.get("instance", {}).get("namespace", None),
        "image_url": config.get("instance", {}).get("image_url", None),
        "env_vars": config.get("instance", {}).get("env_vars", None),
    }
    model_config = InferenceEndpointModelConfig(
        # We only initialize params which have a non default value
        **{k: v for k, v in all_params.items() if v is not None},
    )

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        env_config=env_config,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=override_batch_size,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
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


@app.command(rich_help_panel="Evaluation Backends")
def tgi(
    # === general ===
    model_config_path: Annotated[
        str, Argument(help="Path to model config yaml file. (examples/model_configs/tgi_model.yaml)")
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    use_chat_template: Annotated[
        bool, Option(help="Use chat template for evaluation.", rich_help_panel=HELP_PANNEL_NAME_4)
    ] = False,
    system_prompt: Annotated[
        Optional[str], Option(help="Use system prompt for evaluation.", rich_help_panel=HELP_PANNEL_NAME_4)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = None,
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = CACHE_DIR,
    num_fewshot_seeds: Annotated[
        int, Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    # === saving ===
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = "results",
    push_to_hub: Annotated[
        bool, Option(help="Push results to the huggingface hub.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    push_to_tensorboard: Annotated[
        bool, Option(help="Push results to tensorboard.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    public_run: Annotated[
        bool, Option(help="Push results and details to a public repo.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    results_org: Annotated[
        Optional[str], Option(help="Organization to push results to.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = None,
    save_details: Annotated[
        bool, Option(help="Save detailed, sample per sample, results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = False,
    # === debug ===
    max_samples: Annotated[
        Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = None,
    override_batch_size: Annotated[
        int, Option(help="Override batch size for evaluation.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = -1,
    job_id: Annotated[
        int, Option(help="Optional job id for future refenrence.", rich_help_panel=HELP_PANNEL_NAME_3)
    ] = 0,
):
    """
    Evaluate models using TGI as backend.
    """
    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_config import TGIModelConfig
    from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    env_config = EnvConfig(token=TOKEN, cache_dir=cache_dir)
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )

    # TODO (nathan): better handling of model_args
    parallelism_manager = ParallelismManager.TGI
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)["model"]

    model_config = TGIModelConfig(
        inference_server_address=config["instance"]["inference_server_address"],
        inference_server_auth=config["instance"]["inference_server_auth"],
        model_id=config["instance"]["model_id"],
    )

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        env_config=env_config,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=override_batch_size,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
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
