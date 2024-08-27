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
from typing import Optional

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import htrack, htrack_block
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import NO_NANOTRON_ERROR_MSG, is_nanotron_available
from lighteval.utils.utils import EnvConfig


if not is_nanotron_available():
    raise ImportError(NO_NANOTRON_ERROR_MSG)

from nanotron.config import Config, LightEvalConfig, get_config_from_file


SEED = 1234


@htrack()
def main(
    checkpoint_config_path: str,
    lighteval_config_path: Optional[str] = None,
    cache_dir: Optional[str] = os.getenv("HF_HOME", "/scratch"),
):
    env_config = EnvConfig(token=os.getenv("HF_TOKEN"), cache_dir=cache_dir)

    with htrack_block("Load nanotron config"):
        # Create nanotron config
        if not checkpoint_config_path.endswith(".yaml"):
            raise ValueError("The checkpoint path should point to a YAML file")

        model_config = get_config_from_file(
            checkpoint_config_path,
            config_class=Config,
            model_config_class=None,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )
        if lighteval_config_path:
            lighteval_config = get_config_from_file(lighteval_config_path, config_class=LightEvalConfig)
            model_config.lighteval = lighteval_config
        else:
            lighteval_config = model_config.lighteval

    evaluation_tracker = EvaluationTracker(
        token=os.getenv("HF_TOKEN"),
        output_dir=lighteval_config.logging.local_output_path,
        hub_results_org=lighteval_config.logging.hub_repo_tensorboard,
        tensorboard_metric_prefix=lighteval_config.logging.tensorboard_metric_prefix,
        nanotron_run_info=model_config.general,
    )

    pipeline_parameters = PipelineParameters(
        launcher_type=ParallelismManager.NANOTRON,
        env_config=env_config,
        job_id=os.environ.get("SLURM_JOB_ID", None),
        nanotron_checkpoint_path=checkpoint_config_path,
        dataset_loading_processes=lighteval_config.tasks.dataset_loading_processes,
        custom_tasks_directory=lighteval_config.tasks.custom_tasks,
        override_batch_size=None,
        num_fewshot_seeds=1,
        max_samples=lighteval_config.tasks.max_samples,
        use_chat_template=False,
        system_prompt=None,
    )

    pipeline = Pipeline(
        tasks=lighteval_config.tasks.tasks,
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    pipeline.save_and_push_results()
