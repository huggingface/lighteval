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
from typing import Iterator, TypeAlias
from unittest.mock import patch

import pytest
from huggingface_hub import ChatCompletionInputMessage
from transformers import BatchEncoding

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import (
    Doc,
    Request,
    RequestType,
)


RequestDict: TypeAlias = dict[RequestType, list[Request]]


def test_abstract_model_tokenizer_api(base_model: BaseModel):
    encoded = base_model.tok_encode("Hi there!")
    assert isinstance(encoded, list) and isinstance(encoded[0], int)

    encoded = base_model.tok_encode(ChatCompletionInputMessage(role="user", content="Hi there!"))
    assert encoded == base_model.tok_encode([ChatCompletionInputMessage(role="user", content="Hi there!")])
    assert isinstance(encoded, list) and isinstance(encoded[0], int)

    assert isinstance(
        base_model.tok_encode(["Hi there!", "Hello there!"]),
        BatchEncoding,
    )

    assert isinstance(
        base_model.tok_encode([[ChatCompletionInputMessage(role="user", content="Hi there!")]]), BatchEncoding
    )


class TestBaseModel:
    @pytest.fixture
    def task(self) -> LightevalTask:
        eval_docs = [
            Doc(
                query="Tell me:\n\nHow are you?",
                choices=["Fine, thanks!", "Not bad!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
            Doc(
                query="Tell me:\n\nComment vas-tu?",
                choices=["Ca va! Merci!", "Comme ci, comme ça"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
        ]
        fewshot_docs = [
            Doc(
                query="Tell me:\n\nكيف حالك؟",
                choices=["جيد شكراً!", "ليس سيئًا!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
            Doc(
                query="Tell me:\n\nWie geht es dir?",
                choices=["Gut, danke!", "Nicht schlecht!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
        ]
        task_config = LightevalTaskConfig(
            name="test",
            prompt_function=lambda _: _,
            hf_repo="",
            hf_subset="",
            metric=[Metrics.loglikelihood_acc, Metrics.exact_match, Metrics.byte_perplexity],
            generation_size=5,
            stop_sequence=[],
        )
        task = LightevalTask("test", task_config)
        task._docs = eval_docs
        task._fewshot_docs = fewshot_docs
        return task

    @pytest.mark.parametrize("num_fewshot", [0, 2])
    @pytest.mark.parametrize("use_chat_template", [False, True])
    def test_integration(self, task: LightevalTask, base_model: BaseModel, num_fewshot: int, use_chat_template: bool):
        base_model.use_chat_template = use_chat_template

        evaluation_tracker = EvaluationTracker()
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            use_chat_template=use_chat_template,
            override_batch_size=1,
        )

        with patch("lighteval.pipeline.Pipeline._init_tasks_and_requests"):
            pipeline = Pipeline(
                tasks=f"custom|test|{num_fewshot}|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=base_model,
            )
        task_dict = {"custom|test": task}
        evaluation_tracker.task_config_logger.log(task_dict)
        fewshot_dict = {"custom|test": [(num_fewshot, False)]}
        pipeline.task_names_list = ["custom|test"]
        pipeline.task_dict = task_dict
        pipeline.fewshot_dict = fewshot_dict
        requests, docs = create_requests_from_tasks(
            task_dict=task_dict,
            fewshot_dict=fewshot_dict,
            num_fewshot_seeds=pipeline_params.num_fewshot_seeds,
            lm=base_model,
            max_samples=pipeline_params.max_samples,
            evaluation_tracker=evaluation_tracker,
            use_chat_template=use_chat_template,
            system_prompt=pipeline_params.system_prompt,
        )
        pipeline.requests = requests
        pipeline.docs = docs
        evaluation_tracker.task_config_logger.log(task_dict)

        pipeline.evaluate()
