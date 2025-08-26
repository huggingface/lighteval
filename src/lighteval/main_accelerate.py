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

from typer import Option
from typing_extensions import Annotated

from lighteval.cli_args import (
    HELP_PANEL_NAME_4,
    custom_tasks,
    dataset_loading_processes,
    job_id,
    load_responses_from_details_date_id,
    max_samples,
    model_args,
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


logger = logging.getLogger(__name__)


def accelerate(  # noqa C901
    # === general ===
    model_args: model_args.type,
    tasks: tasks.type,
    # === Common parameters ===
    vision_model: Annotated[
        bool, Option(help="Use vision model for evaluation.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = False,
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
    Evaluate models using accelerate and transformers as backend.
    """
    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.abstract_model import ModelConfig
    from lighteval.models.transformers.adapter_model import AdapterModelConfig
    from lighteval.models.transformers.delta_model import DeltaModelConfig
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval.models.transformers.vlm_transformers_model import VLMTransformersModelConfig
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
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
    )

    if model_args.endswith(".yaml"):
        with open(model_args, "r") as f:
            config = yaml.safe_load(f)["model_parameters"]
    else:
        # We extract the model args
        config: dict = ModelConfig._parse_args(model_args)

    if config.get("delta_weights", False):
        config.pop("delta_weights")
        model_config = DeltaModelConfig(**config)
    elif config.get("adapter_weights", False):
        config.pop("adapter_weights")
        model_config = AdapterModelConfig(**config)
    else:
        if vision_model:
            model_config = VLMTransformersModelConfig(**config)
        else:
            model_config = TransformersModelConfig(**config)

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
