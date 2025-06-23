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
from typing import Optional

import typer
from typer import Argument, Option
from typing_extensions import Annotated

from lighteval.models.custom.custom_model import CustomModelConfig


app = typer.Typer()


HELP_PANNEL_NAME_1 = "Common Parameters"
HELP_PANNEL_NAME_2 = "Logging Parameters"
HELP_PANNEL_NAME_3 = "Debug Parameters"
HELP_PANNEL_NAME_4 = "Modeling Parameters"


@app.command(rich_help_panel="Evaluation Backends")
def custom(
    # === general ===
    model_name: Annotated[str, Argument(help="The model name to evaluate")],
    model_definition_file_path: Annotated[str, Argument(help="The model definition file path to evaluate")],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = None,
    num_fewshot_seeds: Annotated[
        int, Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANNEL_NAME_1)
    ] = 1,
    # === saving ===
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANNEL_NAME_2)
    ] = "results",
    results_path_template: Annotated[
        str | None,
        Option(
            help="Template path for where to save the results, you have access to 3 variables, `output_dir`, `org` and `model`. for example a template can be `'{output_dir}/1234/{org}+{model}'`",
            rich_help_panel=HELP_PANNEL_NAME_2,
        ),
    ] = None,
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
