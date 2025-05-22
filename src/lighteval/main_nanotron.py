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

# flake8: noqa: C901
import os

import yaml
from typer import Option
from typing_extensions import Annotated
from yaml import SafeLoader


HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"


SEED = 1234


def nanotron(
    checkpoint_config_path: Annotated[
        str, Option(help="Path to the nanotron checkpoint YAML or python config file, potentially on s3.")
    ],
    lighteval_config_path: Annotated[str, Option(help="Path to a YAML config to be used for the evaluation.")],
):
    """
    Evaluate models using nanotron as backend.
    """
    from nanotron.config import GeneralArgs, ModelArgs, TokenizerArgs, get_config_from_dict, get_config_from_file

    from lighteval.config.lighteval_config import (
        FullNanotronConfig,
        LightEvalConfig,
    )
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    from lighteval.utils.imports import NO_NANOTRON_ERROR_MSG, is_nanotron_available

    if not is_nanotron_available():
        raise ImportError(NO_NANOTRON_ERROR_MSG)

    # Create nanotron config
    if not checkpoint_config_path.endswith(".yaml"):
        raise ValueError("The checkpoint path should point to a YAML file")

    with open(checkpoint_config_path) as f:
        nanotron_yaml = yaml.load(f, Loader=SafeLoader)

    model_config, tokenizer_config, general_config = [
        get_config_from_dict(
            nanotron_yaml[key],
            config_class=config_class,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )
        for key, config_class in [("model", ModelArgs), ("tokenizer", TokenizerArgs), ("general", GeneralArgs)]
    ]

    # Load lighteval config
    lighteval_config: LightEvalConfig = get_config_from_file(lighteval_config_path, config_class=LightEvalConfig)  # type: ignore
    nanotron_config = FullNanotronConfig(lighteval_config, model_config, tokenizer_config, general_config)

    evaluation_tracker = EvaluationTracker(
        output_dir=lighteval_config.logging.output_dir,
        results_path_template=lighteval_config.logging.results_path_template,
        hub_results_org=lighteval_config.logging.results_org,
        public=lighteval_config.logging.public_run,
        push_to_hub=lighteval_config.logging.push_to_hub,
        push_to_tensorboard=lighteval_config.logging.push_to_tensorboard,
        save_details=lighteval_config.logging.save_details,
        tensorboard_metric_prefix=lighteval_config.logging.tensorboard_metric_prefix,
        nanotron_run_info=nanotron_config.nanotron_general,
    )

    pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.NANOTRON,
        job_id=os.environ.get("SLURM_JOB_ID", 0),
        nanotron_checkpoint_path=checkpoint_config_path,
        dataset_loading_processes=lighteval_config.tasks.dataset_loading_processes,
        custom_tasks_directory=lighteval_config.tasks.custom_tasks,
        num_fewshot_seeds=1,
        max_samples=lighteval_config.tasks.max_samples,
        use_chat_template=False,
        system_prompt=None,
    )

    pipeline = Pipeline(
        tasks=lighteval_config.tasks.tasks,
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=evaluation_tracker,
        model_config=nanotron_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    pipeline.save_and_push_results()
