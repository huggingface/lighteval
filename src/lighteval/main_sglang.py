# MIT License

# Copyright (c) 2024 The SGLang Team

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

from typer import Argument, Option
from typing_extensions import Annotated


HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


def sglang(
    # === general ===
    model_args: Annotated[
        str,
        Argument(
            help="Model arguments in the form key1=value1,key2=value2,... or path to yaml config file (see examples/model_configs/transformers_model.yaml)"
        ),
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANEL_NAME_1)
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
    results_path_template: Annotated[
        str | None,
        Option(
            help="Template path for where to save the results, you have access to 3 variables, `output_dir`, `org` and `model`. for example a template can be `'{output_dir}/1234/{org}+{model}'`",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = None,
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
    wandb: Annotated[
        bool,
        Option(
            help="Push results to wandb. This will only work if you have wandb installed and logged in. We use env variable to configure wandb. see here: https://docs.wandb.ai/guides/track/environment-variables/",
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
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
    Evaluate models using sglang as backend.
    """
    import yaml

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.sglang.sglang_model import SGLangModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        wandb=wandb,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.SGLANG,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
    )

    if model_args.endswith(".yaml"):
        with open(model_args, "r") as f:
            metric_options = yaml.safe_load(f).get("metric_options", {})
        model_config = SGLangModelConfig.from_path(model_args)
    else:
        metric_options = {}
        model_config = SGLangModelConfig.from_args(model_args)

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
