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


import typer
from typer import Argument, Option
from typing_extensions import Annotated

from lighteval.cli_args import (
    DEFAULT_VALUES,
    HELP_PANEL_NAME_4,
    CustomTasks,
    DatasetLoadingProcesses,
    JobId,
    LoadResponsesFromDetailsDateId,
    MaxSamples,
    NumFewshotSeeds,
    OutputDir,
    PublicRun,
    PushToHub,
    PushToTensorboard,
    ReasoningTags,
    RemoveReasoningTags,
    ResultsOrg,
    ResultsPathTemplate,
    SaveDetails,
    Tasks,
    Wandb,
)


app = typer.Typer()


@app.command(rich_help_panel="Evaluation Backends")
def inference_endpoint(
    # === general ===
    model_config_path: Annotated[
        str, Argument(help="Path to model config yaml file. (examples/model_configs/endpoint_model.yaml)")
    ],
    tasks: Tasks,
    free_endpoint: Annotated[
        bool,
        Option(
            help="Use serverless free endpoints instead of spinning up your own inference endpoint.",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = False,
    # === Common parameters ===
    dataset_loading_processes: DatasetLoadingProcesses = DEFAULT_VALUES["dataset_loading_processes"],
    custom_tasks: CustomTasks = DEFAULT_VALUES["custom_tasks"],
    num_fewshot_seeds: NumFewshotSeeds = DEFAULT_VALUES["num_fewshot_seeds"],
    load_responses_from_details_date_id: LoadResponsesFromDetailsDateId = DEFAULT_VALUES[
        "load_responses_from_details_date_id"
    ],
    remove_reasoning_tags: RemoveReasoningTags = DEFAULT_VALUES["remove_reasoning_tags"],
    reasoning_tags: ReasoningTags = DEFAULT_VALUES["reasoning_tags"],
    # === saving ===
    output_dir: OutputDir = DEFAULT_VALUES["output_dir"],
    results_path_template: ResultsPathTemplate = DEFAULT_VALUES["results_path_template"],
    push_to_hub: PushToHub = DEFAULT_VALUES["push_to_hub"],
    push_to_tensorboard: PushToTensorboard = DEFAULT_VALUES["push_to_tensorboard"],
    public_run: PublicRun = DEFAULT_VALUES["public_run"],
    results_org: ResultsOrg = DEFAULT_VALUES["results_org"],
    save_details: SaveDetails = DEFAULT_VALUES["save_details"],
    wandb: Wandb = DEFAULT_VALUES["wandb"],
    # === debug ===
    max_samples: MaxSamples = DEFAULT_VALUES["max_samples"],
    job_id: JobId = DEFAULT_VALUES["job_id"],
):
    """
    Evaluate models using inference-endpoints as backend.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.endpoints.endpoint_model import InferenceEndpointModelConfig, ServerlessEndpointModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        use_wandb=wandb,
    )

    parallelism_manager = ParallelismManager.NONE  # since we're using inference endpoints in remote

    if free_endpoint:
        model_config = ServerlessEndpointModelConfig.from_path(model_config_path)
    else:
        model_config = InferenceEndpointModelConfig.from_path(model_config_path)

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
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
    tasks: Tasks,
    # === Common parameters ===
    dataset_loading_processes: DatasetLoadingProcesses = DEFAULT_VALUES["dataset_loading_processes"],
    custom_tasks: CustomTasks = DEFAULT_VALUES["custom_tasks"],
    num_fewshot_seeds: NumFewshotSeeds = DEFAULT_VALUES["num_fewshot_seeds"],
    load_responses_from_details_date_id: LoadResponsesFromDetailsDateId = DEFAULT_VALUES[
        "load_responses_from_details_date_id"
    ],
    remove_reasoning_tags: RemoveReasoningTags = DEFAULT_VALUES["remove_reasoning_tags"],
    reasoning_tags: ReasoningTags = DEFAULT_VALUES["reasoning_tags"],
    # === saving ===
    output_dir: OutputDir = DEFAULT_VALUES["output_dir"],
    results_path_template: ResultsPathTemplate = DEFAULT_VALUES["results_path_template"],
    push_to_hub: PushToHub = DEFAULT_VALUES["push_to_hub"],
    push_to_tensorboard: PushToTensorboard = DEFAULT_VALUES["push_to_tensorboard"],
    public_run: PublicRun = DEFAULT_VALUES["public_run"],
    results_org: ResultsOrg = DEFAULT_VALUES["results_org"],
    save_details: SaveDetails = DEFAULT_VALUES["save_details"],
    wandb: Wandb = DEFAULT_VALUES["wandb"],
    # === debug ===
    max_samples: MaxSamples = DEFAULT_VALUES["max_samples"],
    job_id: JobId = DEFAULT_VALUES["job_id"],
):
    """
    Evaluate models using TGI as backend.
    """
    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.endpoints.tgi_model import TGIModelConfig
    from lighteval.models.model_input import GenerationParameters
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        use_wandb=wandb,
    )

    parallelism_manager = ParallelismManager.TGI

    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    generation_parameters = GenerationParameters(**config.get("generation", {}))
    model_config = TGIModelConfig(**config["model"], generation_parameters=generation_parameters)

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
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
def litellm(
    # === general ===
    model_args: Annotated[
        str,
        Argument(
            help="config file path for the litellm model, or a comma separated string of model args (model_name={},base_url={},provider={})"
        ),
    ],
    tasks: Tasks,
    # === Common parameters ===
    dataset_loading_processes: DatasetLoadingProcesses = DEFAULT_VALUES["dataset_loading_processes"],
    custom_tasks: CustomTasks = DEFAULT_VALUES["custom_tasks"],
    num_fewshot_seeds: NumFewshotSeeds = DEFAULT_VALUES["num_fewshot_seeds"],
    load_responses_from_details_date_id: LoadResponsesFromDetailsDateId = DEFAULT_VALUES[
        "load_responses_from_details_date_id"
    ],
    remove_reasoning_tags: RemoveReasoningTags = DEFAULT_VALUES["remove_reasoning_tags"],
    reasoning_tags: ReasoningTags = DEFAULT_VALUES["reasoning_tags"],
    # === saving ===
    output_dir: OutputDir = DEFAULT_VALUES["output_dir"],
    results_path_template: ResultsPathTemplate = DEFAULT_VALUES["results_path_template"],
    push_to_hub: PushToHub = DEFAULT_VALUES["push_to_hub"],
    push_to_tensorboard: PushToTensorboard = DEFAULT_VALUES["push_to_tensorboard"],
    public_run: PublicRun = DEFAULT_VALUES["public_run"],
    results_org: ResultsOrg = DEFAULT_VALUES["results_org"],
    save_details: SaveDetails = DEFAULT_VALUES["save_details"],
    wandb: Wandb = DEFAULT_VALUES["wandb"],
    # === debug ===
    max_samples: MaxSamples = DEFAULT_VALUES["max_samples"],
    job_id: JobId = DEFAULT_VALUES["job_id"],
):
    """
    Evaluate models using LiteLLM as backend.
    """

    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        use_wandb=wandb,
    )

    parallelism_manager = ParallelismManager.NONE

    if model_args.endswith(".yaml"):
        with open(model_args, "r") as f:
            config = yaml.safe_load(f)
        metric_options = config.get("metric_options", {})
        model_config = LiteLLMModelConfig.from_path(model_args)
    else:
        metric_options = None
        model_config = LiteLLMModelConfig.from_args(model_args)

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
    )
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


@app.command(rich_help_panel="Evaluation Backends")
def inference_providers(
    # === general ===
    model_args: Annotated[
        str,
        Argument(
            help="config file path for the inference provider model, or a comma separated string of model args (model_name={},provider={},generation={temperature: 0.6})"
        ),
    ],
    tasks: Tasks,
    # === Common parameters ===
    dataset_loading_processes: DatasetLoadingProcesses = DEFAULT_VALUES["dataset_loading_processes"],
    custom_tasks: CustomTasks = DEFAULT_VALUES["custom_tasks"],
    num_fewshot_seeds: NumFewshotSeeds = DEFAULT_VALUES["num_fewshot_seeds"],
    # === saving ===
    output_dir: OutputDir = DEFAULT_VALUES["output_dir"],
    results_path_template: ResultsPathTemplate = DEFAULT_VALUES["results_path_template"],
    push_to_hub: PushToHub = DEFAULT_VALUES["push_to_hub"],
    push_to_tensorboard: PushToTensorboard = DEFAULT_VALUES["push_to_tensorboard"],
    public_run: PublicRun = DEFAULT_VALUES["public_run"],
    results_org: ResultsOrg = DEFAULT_VALUES["results_org"],
    save_details: SaveDetails = DEFAULT_VALUES["save_details"],
    wandb: Wandb = DEFAULT_VALUES["wandb"],
    remove_reasoning_tags: RemoveReasoningTags = DEFAULT_VALUES["remove_reasoning_tags"],
    reasoning_tags: ReasoningTags = DEFAULT_VALUES["reasoning_tags"],
    # === debug ===
    max_samples: MaxSamples = DEFAULT_VALUES["max_samples"],
    job_id: JobId = DEFAULT_VALUES["job_id"],
):
    """
    Evaluate models using HuggingFace's inference providers as backend.
    """

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.endpoints.inference_providers_model import (
        InferenceProvidersModelConfig,
    )
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        use_wandb=wandb,
    )

    parallelism_manager = ParallelismManager.NONE

    if model_args.endswith(".yaml"):
        model_config = InferenceProvidersModelConfig.from_path(model_args)
    else:
        model_config = InferenceProvidersModelConfig.from_args(model_args)

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=None,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
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
