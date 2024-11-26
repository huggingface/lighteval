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

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import htrack
from lighteval.models.model_config import VLLMModelConfig
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters


@htrack()
def main(args):
    TOKEN = os.getenv("HF_TOKEN")

    env_config = EnvConfig(token=TOKEN, cache_dir=args.cache_dir)

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=args.save_details,
        push_to_hub=args.push_to_hub,
        push_to_tensorboard=args.push_to_tensorboard,
        public=args.public_run,
        hub_results_org=args.results_org,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        env_config=env_config,
        job_id=args.job_id,
        dataset_loading_processes=args.dataset_loading_processes,
        custom_tasks_directory=args.custom_tasks,
        override_batch_size=args.override_batch_size,
        num_fewshot_seeds=args.num_fewshot_seeds,
        max_samples=args.max_samples,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )

    model_args: dict = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in args.model_args.split(",")}
    model_config = VLLMModelConfig(**model_args)

    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    results = pipeline.get_results()

    pipeline.save_and_push_results()

    return results
