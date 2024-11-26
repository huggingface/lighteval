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

import yaml

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import htrack
from lighteval.models.model_config import (
    InferenceEndpointModelConfig,
    InferenceModelConfig,
    OpenAIModelConfig,
    TGIModelConfig,
)
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters


TOKEN = os.getenv("HF_TOKEN")


@htrack()
def main(args):
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
        launcher_type=ParallelismManager.ACCELERATE,
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

    # TODO (nathan): better handling of model_args

    if args.provider == "openai":
        model_args: dict = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in args.model_args.split(",")}
        model_config = OpenAIModelConfig(**model_args)
    elif args.provider == "tgi":
        with open(args.model_config_path, "r") as f:
            config = yaml.safe_load(f)["model"]
        model_config = TGIModelConfig(
            inference_server_address=config["instance"]["inference_server_address"],
            inference_server_auth=config["instance"]["inference_server_auth"],
            model_id=config["instance"]["model_id"],
        )
    elif args.provider == "inference_endpoints":
        with open(args.model_config_path, "r") as f:
            config = yaml.safe_load(f)["model"]
        reuse_existing_endpoint = config["base_params"].get("reuse_existing", None)
        complete_config_endpoint = all(
            val not in [None, ""]
            for key, val in config.get("instance", {}).items()
            if key not in InferenceEndpointModelConfig.nullable_keys()
        )
        if reuse_existing_endpoint or complete_config_endpoint:
            model_config = InferenceEndpointModelConfig(
                name=config["base_params"]["endpoint_name"].replace(".", "-").lower(),
                repository=config["base_params"]["model"],
                model_dtype=config["base_params"]["dtype"],
                revision=config["base_params"]["revision"] or "main",
                should_reuse_existing=reuse_existing_endpoint,
                accelerator=config["instance"]["accelerator"],
                region=config["instance"]["region"],
                vendor=config["instance"]["vendor"],
                instance_size=config["instance"]["instance_size"],
                instance_type=config["instance"]["instance_type"],
                namespace=config["instance"]["namespace"],
                image_url=config["instance"].get("image_url", None),
                env_vars=config["instance"].get("env_vars", None),
            )
        else:
            model_config = InferenceModelConfig(model=config["base_params"]["endpoint_name"])
    else:
        raise ValueError(f"Unsupported provider for lighteval endpoint: {args.provider}")

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
