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
from typer import Argument
from typing_extensions import Annotated

from lighteval.cli_args import (
    DEFAULT_VALUES,
    CustomTasks,
    DatasetLoadingProcesses,
    JobId,
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
)
from lighteval.models.custom.custom_model import CustomModelConfig


app = typer.Typer()


@app.command(rich_help_panel="Evaluation Backends")
def custom(
    # === general ===
    model_name: Annotated[str, Argument(help="The model name to evaluate")],
    model_definition_file_path: Annotated[str, Argument(help="The model definition file path to evaluate")],
    tasks: Tasks,
    # === Common parameters ===
    dataset_loading_processes: DatasetLoadingProcesses = DEFAULT_VALUES["dataset_loading_processes"],
    custom_tasks: CustomTasks = DEFAULT_VALUES["custom_tasks"],
    num_fewshot_seeds: NumFewshotSeeds = DEFAULT_VALUES["num_fewshot_seeds"],
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
    # === debug ===
    max_samples: MaxSamples = DEFAULT_VALUES["max_samples"],
    job_id: JobId = DEFAULT_VALUES["job_id"],
):
    """
    Evaluate custom models (can be anything).
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )

    parallelism_manager = ParallelismManager.CUSTOM
    model_config = CustomModelConfig(model_name=model_name, model_definition_file_path=model_definition_file_path)

    pipeline_params = PipelineParameters(
        launcher_type=parallelism_manager,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
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
