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
import random
import time
from collections import defaultdict
from typing import Iterator, TypeAlias

import docker
import pytest
import requests
import torch
from huggingface_hub import ChatCompletionInputMessage, ChatCompletionOutputMessage

from lighteval.evaluator import EvaluationTracker, evaluate
from lighteval.metrics.metrics import Metrics
from lighteval.models.tgi_model import ModelClient as TGIModel
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import (
    Doc,
    Request,
    RequestType,
)


TOKEN = os.environ.get("HF_TOKEN")
CACHE_PATH = os.getenv("HF_HOME", ".")


@pytest.fixture(scope="module")
def tgi_model() -> Iterator[TGIModel]:
    client = docker.from_env()
    port = random.randint(8000, 9000)
    container = client.containers.run(
        "ghcr.io/huggingface/text-generation-inference:2.2.0",
        command=[
            "--model-id",
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "--dtype",
            "float16",
        ],
        detach=True,
        name="lighteval-tgi-model-test",
        auto_remove=True,
        ports={"80/tcp": port},
    )
    address = f"http://localhost:{port}"
    for _ in range(30):
        try:
            if requests.get(f"{address}/health"):
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Couldn't setup TGI server.")
    model = TGIModel(address)
    yield model
    container.stop()
    container.wait()
    model.cleanup()


RequestDict: TypeAlias = dict[RequestType, list[Request]]


class TestEndpointModel:
    @pytest.fixture
    def task(self) -> LightevalTask:
        eval_docs = [
            Doc(query="How are you?", choices=["Fine, thanks!", "Not bad!"], instruction="Tell me:\n\n", gold_index=0),
            Doc(
                query="Comment vas-tu?",
                choices=["Ca va! Merci!", "Comme ci, comme ça"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
        ]
        fewshot_docs = [
            Doc(query="كيف حالك؟", choices=["جيد شكراً!", "ليس سيئًا!"], instruction="Tell me:\n\n", gold_index=0),
            Doc(
                query="Wie geht es dir?",
                choices=["Gut, danke!", "Nicht schlecht!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
        ]
        task_config = LightevalTaskConfig(
            "test", lambda _: _, "", "", [Metrics.loglikelihood_acc, Metrics.exact_match, Metrics.byte_perplexity]
        )
        task = LightevalTask("test", task_config)
        task._docs = eval_docs
        task._fewshot_docs = fewshot_docs
        return task

    @pytest.fixture
    def zero_shot_request_dict(self, task: LightevalTask) -> RequestDict:
        result = defaultdict(list)
        for i, doc in enumerate(task.eval_docs()):
            if i % 2 == 0:
                context = [ChatCompletionInputMessage(role="user", content=doc.query)]
            else:
                context = doc.query
            doc_result = task.construct_requests(doc, context, f"{i}_0", "custom|test|0")
            for req_type in doc_result:
                result[req_type].extend(doc_result[req_type])
        return result

    def test_model_tokenizer_api(self, zero_shot_request_dict: RequestDict, tgi_model: TGIModel):
        assert tgi_model.tok_encode(ChatCompletionInputMessage("user", "Hi there!")) == tgi_model.tok_encode(
            [ChatCompletionInputMessage("user", "Hi there!")]
        )
        assert isinstance(
            tgi_model.tok_encode([req.context for req in zero_shot_request_dict[RequestType.GREEDY_UNTIL]]),
            torch.Tensor,
        )

    def test_greedy_until(self, zero_shot_request_dict: RequestDict, tgi_model: TGIModel):
        returns = tgi_model.greedy_until(zero_shot_request_dict[RequestType.GREEDY_UNTIL])
        assert len(returns) == 6
        assert all(isinstance(r.result, ChatCompletionOutputMessage) and r.result.content for r in returns[:3])
        assert None not in (returns[2].logits, returns[5].logits)

    def test_loglikelihood(self, zero_shot_request_dict: RequestDict, tgi_model: TGIModel):
        returns = tgi_model.loglikelihood(zero_shot_request_dict[RequestType.LOGLIKELIHOOD])
        assert len(returns) == 4
        assert all(r.result[0] is not None for r in returns)

        returns = tgi_model.loglikelihood_rolling(zero_shot_request_dict[RequestType.LOGLIKELIHOOD_ROLLING])
        assert len(returns) == 2
        assert all(r.result[0] is not None for r in returns)

    @pytest.mark.parametrize("num_fewshot", [0, 2])
    @pytest.mark.parametrize("use_chat_template", [False, True])
    def test_integration(self, task: LightevalTask, tgi_model: TGIModel, num_fewshot: int, use_chat_template: bool):
        evaluation_tracker = EvaluationTracker()
        task_dict = {"custom|test": task}
        evaluation_tracker.task_config_logger.log(task_dict)
        requests_dict, docs = create_requests_from_tasks(
            task_dict=task_dict,
            fewshot_dict={"custom|test": [(num_fewshot, False)]},
            num_fewshot_seeds=0,
            lm=tgi_model,
            max_samples=1,
            evaluation_tracker=evaluation_tracker,
            use_chat_template=use_chat_template,
        )

        evaluation_tracker = evaluate(
            lm=tgi_model,
            requests_dict=requests_dict,
            docs=docs,
            task_dict=task_dict,
            override_bs=1,
            evaluation_tracker=evaluation_tracker,
        )
        evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict)
        evaluation_tracker.details_logger.aggregate()
        evaluation_tracker.generate_final_dict()
