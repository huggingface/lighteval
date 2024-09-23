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
import time
from collections import defaultdict
from typing import Iterator, TypeAlias
from unittest.mock import patch

import docker
import docker.errors
import pytest
import requests
import torch
from huggingface_hub import ChatCompletionInputMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.tgi_model import ModelClient as TGIModel
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import (
    Doc,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    Request,
    RequestType,
)


@pytest.fixture(scope="module")
def tgi_model() -> Iterator[TGIModel]:
    client = docker.from_env()

    try:
        container = client.containers.get("lighteval-tgi-model-test")
        port = container.ports["80/tcp"][0]["HostPort"]
    except docker.errors.NotFound:
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
            auto_remove=False,
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
    # container.stop()
    # container.wait()
    # container.remove()
    model.cleanup()


@pytest.fixture(scope="module")
def reference_model_tokenizer() -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    return model, tokenizer


RequestDict: TypeAlias = dict[RequestType, list[Request]]


class TestEndpointModel:
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

    def test_greedy_until(
        self,
        reference_model_tokenizer: tuple[LlamaForCausalLM, PreTrainedTokenizerFast],
        zero_shot_request_dict: RequestDict,
        tgi_model: TGIModel,
    ):
        requests = zero_shot_request_dict[RequestType.GREEDY_UNTIL]
        returns = tgi_model.greedy_until(requests)
        model, tokenizer = reference_model_tokenizer
        assert len(returns) == 2
        for req, res in zip(requests, returns):
            is_chat = not isinstance(req.context, str)
            tokenized_context = (
                tokenizer.apply_chat_template(req.context, return_tensors="pt")
                if is_chat
                else tokenizer(req.context, return_tensors="pt")["input_ids"]
            )
            ref_context_continuaiton = model.generate(
                tokenized_context,
                tokenizer=tokenizer,
                stop_strings=req.stop_sequence,
                max_new_tokens=req.generation_size,
            )[0].tolist()
            continuation = tokenizer.decode(ref_context_continuaiton)[
                len(tokenizer.decode(tokenized_context[0].tolist())) :
            ]
            assert continuation == res.result

    def test_loglikelihood(
        self,
        reference_model_tokenizer: tuple[LlamaForCausalLM, PreTrainedTokenizerFast],
        zero_shot_request_dict: RequestDict,
        tgi_model: TGIModel,
    ):
        requests: list[LoglikelihoodRequest] = zero_shot_request_dict[RequestType.LOGLIKELIHOOD]

        # https://github.com/huggingface/text-generation-inference/issues/2502
        with patch.object(tgi_model, "use_async", False):
            returns = tgi_model.loglikelihood(requests)

        model, tokenizer = reference_model_tokenizer
        assert len(returns) == 4
        for req, res in zip(requests, returns):
            is_chat = not isinstance(req.context, str)
            sequence = (
                req.context + [ChatCompletionInputMessage(role="assistant", content=req.choice)]
                if is_chat
                else req.context + req.choice
            )
            tokenized_sequence = (
                tokenizer.apply_chat_template(sequence, return_tensors="pt")
                if is_chat
                else tokenizer(sequence, return_tensors="pt")["input_ids"]
            )

            output = model.generate(
                tokenized_sequence, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True
            )
            with torch.no_grad():
                logprobs = torch.log_softmax(model.lm_head(output.hidden_states[0][-1]), dim=-1)
            logprobs = logprobs.gather(dim=-1, index=tokenized_sequence[:, 1:].unsqueeze(-1))
            context_length = (
                len(tokenizer.apply_chat_template(req.context)) if is_chat else len(tokenizer.encode(req.context))
            )
            continuation_logprob = logprobs[:, context_length - 1 :].sum()

            tokenized_choice = tokenized_sequence[:, context_length:]
            assert tokenized_choice[0].tolist() == res.input_tokens
            assert torch.allclose(torch.tensor(res.result[0]), continuation_logprob)

    def test_loglikelihood_rolling(
        self,
        reference_model_tokenizer: tuple[LlamaForCausalLM, PreTrainedTokenizerFast],
        zero_shot_request_dict: RequestDict,
        tgi_model: TGIModel,
    ):
        model, tokenizer = reference_model_tokenizer
        requests: list[LoglikelihoodRollingRequest] = zero_shot_request_dict[RequestType.LOGLIKELIHOOD_ROLLING]
        returns = tgi_model.loglikelihood_rolling(zero_shot_request_dict[RequestType.LOGLIKELIHOOD_ROLLING])
        assert len(returns) == 2
        for req, res in zip(requests, returns):
            is_chat = not isinstance(req.context, str)
            tokenized_context = (
                tokenizer.apply_chat_template(req.context, return_tensors="pt")
                if is_chat
                else tokenizer(req.context, return_tensors="pt")["input_ids"]
            )
            output = model.generate(
                tokenized_context, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True
            )
            with torch.no_grad():
                logprobs = torch.log_softmax(model.lm_head(output.hidden_states[0][-1]), dim=-1)
            logprob = logprobs.gather(dim=-1, index=tokenized_context[:, 1:].unsqueeze(-1)).sum()

            assert tokenized_context[0, 1:].tolist() == res.input_tokens
            assert torch.allclose(torch.tensor(res.result), logprob)

    @pytest.mark.parametrize("num_fewshot", [0, 2])
    @pytest.mark.parametrize("use_chat_template", [False, True])
    def test_integration(self, task: LightevalTask, tgi_model: TGIModel, num_fewshot: int, use_chat_template: bool):
        evaluation_tracker = EvaluationTracker()
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            use_chat_template=use_chat_template,
        )

        with patch("lighteval.pipeline.Pipeline._init_tasks_and_requests"):
            pipeline = Pipeline(
                tasks=f"custom|test|{num_fewshot}|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=tgi_model,
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
            lm=tgi_model,
            max_samples=pipeline_params.max_samples,
            evaluation_tracker=evaluation_tracker,
            use_chat_template=use_chat_template,
            system_prompt=pipeline_params.system_prompt,
        )
        pipeline.requests = requests
        pipeline.docs = docs
        evaluation_tracker.task_config_logger.log(task_dict)

        pipeline.evaluate()
