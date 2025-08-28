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
    HELP_PANEL_NAME_4,
    custom_tasks,
    dataset_loading_processes,
    job_id,
    load_responses_from_details_date_id,
    max_samples,
    num_fewshot_seeds,
    output_dir,
    public_run,
    push_to_hub,
    push_to_tensorboard,
    reasoning_tags,
    remove_reasoning_tags,
    results_org,
    results_path_template,
    save_details,
    tasks,
    wandb,
)


app = typer.Typer()


@app.command(rich_help_panel="Evaluation Backends")
def inference_endpoint(
    # === general ===
    model_config_path: Annotated[
        str, Argument(help="Path to model config yaml file. (examples/model_configs/endpoint_model.yaml)")
    ],
    tasks: tasks.type,
    free_endpoint: Annotated[
        bool,
        Option(
            help="Use serverless free endpoints instead of spinning up your own inference endpoint.",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = False,
    # === Common parameters ===
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    # === saving ===
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = save_details.default,
    wandb: wandb.type = wandb.default,
    # === debug ===
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
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
    tasks: tasks.type,
    # === Common parameters ===
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    # === saving ===
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = save_details.default,
    wandb: wandb.type = wandb.default,
    # === debug ===
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
):
    """
    Evaluate models using TGI as backend.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.endpoints.tgi_model import TGIModelConfig
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

    model_config = TGIModelConfig.from_path(model_config_path)

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
    tasks: tasks.type,
    # === Common parameters ===
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    # === saving ===
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = save_details.default,
    wandb: wandb.type = wandb.default,
    # === debug ===
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
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
    tasks: tasks.type,
    # === Common parameters ===
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    # === saving ===
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = save_details.default,
    wandb: wandb.type = wandb.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    # === debug ===
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
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
