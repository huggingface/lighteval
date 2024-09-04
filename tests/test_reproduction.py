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

from unittest.mock import patch

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.models.model_loader import load_model
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import Doc


def run_evaluation() -> dict:
    task_config = LightevalTaskConfig(
        name="test",
        prompt_function=lambda _: _,
        hf_repo="",
        hf_subset="",
        metric=[Metrics.loglikelihood_acc],
        generation_size=5,
        stop_sequence=[],
    )
    task = LightevalTask("test", task_config)
    task._docs = [
        Doc(
            query="Tell me:\n\nHow are you?",
            choices=["Fine, thanks!", "Not bad!"],
            instruction="Tell me:\n\n",
            gold_index=0,
        ),
    ]
    task._fewshot_docs = []

    model_config = BaseModelConfig("hf-internal-testing/tiny-random-LlamaForCausalLM")
    model = load_model(config=model_config, env_config=EnvConfig(cache_dir="."))

    evaluation_tracker = EvaluationTracker()
    pipeline_params = PipelineParameters(launcher_type=ParallelismManager.NONE, override_batch_size=0)

    with patch("lighteval.pipeline.Pipeline._init_tasks_and_requests"):
        pipeline = Pipeline(
            tasks="custom|test|0|0",
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=model,
        )
    task_dict = {"custom|test": task}
    evaluation_tracker.task_config_logger.log(task_dict)
    fewshot_dict = {"custom|test": [(0, False)]}
    pipeline.task_names_list = ["custom|test"]
    pipeline.task_dict = task_dict
    pipeline.fewshot_dict = fewshot_dict
    requests, docs = create_requests_from_tasks(
        task_dict=task_dict,
        fewshot_dict=fewshot_dict,
        num_fewshot_seeds=pipeline_params.num_fewshot_seeds,
        lm=model,
        max_samples=pipeline_params.max_samples,
        evaluation_tracker=evaluation_tracker,
        use_chat_template=False,
        system_prompt=pipeline_params.system_prompt,
    )
    pipeline.requests = requests
    pipeline.docs = docs
    evaluation_tracker.task_config_logger.log(task_dict)

    pipeline.evaluate()
    return pipeline.get_results()


def test_reproduction():
    result_1 = run_evaluation()
    del result_1["config_general"]["start_time"]
    del result_1["config_general"]["end_time"]
    del result_1["config_general"]["total_evaluation_time_secondes"]
    result_2 = run_evaluation()
    del result_2["config_general"]["start_time"]
    del result_2["config_general"]["end_time"]
    del result_2["config_general"]["total_evaluation_time_secondes"]
    assert result_2["config_general"] == result_1["config_general"]
    assert result_2["results"] == result_1["results"]
    assert result_2["summary_general"] == result_1["summary_general"]
    assert result_2["versions"] == result_1["versions"]
    assert result_2["summary_tasks"] == result_1["summary_tasks"]
