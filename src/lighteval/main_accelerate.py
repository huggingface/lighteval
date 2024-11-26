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
from typing import Optional

import typer
from typer import Argument, Option
from typing_extensions import Annotated


app = typer.Typer()

TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")


@app.command()
def accelerate(
    # === general ===
    model_args: Annotated[str, Argument(help="Model arguments in the form key1=value1,key2=value2,...")],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === Common parameters ===
    output_dir: Annotated[str, Option(help="Output directory for evaluation results.")] = "results",
    use_chat_template: Annotated[bool, Option(help="Use chat template for evaluation.")] = False,
    system_prompt: Annotated[Optional[str], Option(help="Use system prompt for evaluation.")] = None,
    dataset_loading_processes: Annotated[int, Option(help="Number of processes to use for dataset loading.")] = 1,
    custom_tasks: Annotated[Optional[str], Option(help="Path to custom tasks directory.")] = None,
    cache_dir: Annotated[str, Option(help="Cache directory for datasets and models.")] = CACHE_DIR,
    num_fewshot_seeds: Annotated[int, Option(help="Number of seeds to use for few-shot evaluation.")] = 1,
    # === saving ===
    push_to_hub: Annotated[bool, Option(help="Push results to the huggingface hub.")] = False,
    push_to_tensorboard: Annotated[bool, Option(help="Push results to tensorboard.")] = False,
    public_run: Annotated[bool, Option(help="Push results and details to a public repo.")] = False,
    results_org: Annotated[Optional[str], Option(help="Organization to push results to.")] = None,
    save_details: Annotated[bool, Option(help="Save detailed, sample per sample, results.")] = False,
    # === debug ===
    max_samples: Annotated[Optional[int], Option(help="Maximum number of samples to evaluate on.")] = None,
    override_batch_size: Annotated[int, Option(help="Override batch size for evaluation.")] = -1,
    job_id: Annotated[int, Option(help="Optional job id for future refenrence.")] = 0,
):
    """
    Evaluate models using accelerate and transformers as backend.
    """
    from datetime import timedelta

    from accelerate import Accelerator, InitProcessGroupKwargs

    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.model_config import BaseModelConfig
    from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])

    env_config = EnvConfig(token=TOKEN, cache_dir=cache_dir)
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=env_config,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        override_batch_size=override_batch_size,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )

    # TODO (nathan): better handling of model_args
    model_args: dict = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in model_args.split(",")}
    model_args["accelerator"] = accelerator
    model_args["use_chat_template"] = use_chat_template
    model_args["compile"] = bool(model_args["compile"]) if "compile" in model_args else False
    model_config = BaseModelConfig(**model_args)

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
