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

import random
import re
from unittest.mock import patch

import pytest

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.models.model_output import RegexAnswerExtractor
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import (
    Doc,
)


def test_answer_extractor():
    # MMLU-Pro
    extractor = RegexAnswerExtractor(
        [re.compile(r"answer is \(?\(([A-J])\)?\)"), re.compile(r"\.*\[aA\]nswer:\s*\(([A-J])\)")], fallback="random"
    )

    assert extractor("answer is (C)", ["A", "B", "C", "D"]) == "C"

    random.seed(41)
    fallback_choice = random.choice(["A", "B", "C", "D"])
    random.seed(41)
    assert extractor("answer is (F)", ["A", "B", "C", "D"]) == fallback_choice

    extractor.fallback = "keep"
    assert extractor("I don't know", ["A", "B", "C", "D"]) == "I don't know"

    extractor.fallback = 0
    assert extractor("I don't know", ["A", "B", "C", "D"]) == "A"

    extractor.fallback = "empty_string"
    assert extractor("I don't know", ["A", "B", "C", "D"]) == ""


@pytest.fixture(scope="module")
def base_model() -> BaseModel:
    config = BaseModelConfig("hf-internal-testing/tiny-random-LlamaForCausalLM")
    return BaseModel(config, EnvConfig("."))


@pytest.fixture
def task() -> LightevalTask:
    eval_docs = [
        Doc(
            query="Tell me:\n\nHow many eyes do you have?",
            choices=["2", "3"],
            instruction="Tell me:\n\n",
            gold_index=0,
        ),
        Doc(
            query="Tell me:\n\nHow many hands do we have?",
            choices=["2", "3"],
            instruction="Tell me:\n\n",
            gold_index=0,
        ),
    ]
    task_config = LightevalTaskConfig(
        name="test",
        prompt_function=lambda _: _,
        hf_repo="",
        hf_subset="",
        metric=[Metrics.exact_match],
        answer_extractor=RegexAnswerExtractor([r"\w", r"\d"]),
        generation_size=1,
        stop_sequence=[],
    )
    task = LightevalTask("test", task_config)
    task._docs = eval_docs
    return task


def test_integration(task: LightevalTask, base_model: BaseModel):
    evaluation_tracker = EvaluationTracker(".")
    pipeline_params = PipelineParameters(
        env_config=EnvConfig("."),
        launcher_type=ParallelismManager.NONE,
        override_batch_size=0,
        use_chat_template=False,
    )
    with patch("lighteval.pipeline.Pipeline._init_tasks_and_requests"):
        pipeline = Pipeline(
            tasks="custom|test|0|0",
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=base_model,
        )
    task_dict = {"custom|test": task}
    evaluation_tracker.task_config_logger.log(task_dict)
    pipeline.task_dict = task_dict
    pipeline.task_names_list = ["custom|test"]
    requests_dict, docs = create_requests_from_tasks(
        task_dict=task_dict,
        fewshot_dict={"custom|test": [(0, False)]},
        num_fewshot_seeds=pipeline_params.num_fewshot_seeds,
        lm=base_model,
        max_samples=pipeline_params.max_samples,
        evaluation_tracker=evaluation_tracker,
        use_chat_template=False,
        system_prompt=pipeline_params.system_prompt,
    )
    pipeline.requests = requests_dict
    pipeline.docs = docs
    evaluation_tracker.task_config_logger.log(task_dict)
    pipeline.evaluate()
