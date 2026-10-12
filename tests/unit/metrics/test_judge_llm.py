# MIT License

# Copyright (c) 2026 The HuggingFace Team

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

import pytest

from lighteval.metrics.metrics_sample import JudgeLLM


@pytest.mark.parametrize("judge_backend", ["transformers", "vllm"])
def test_judge_llm_local_backends_construct_without_hub_lookup(judge_backend):
    # Building the metric must not query the Hub. The removed check called
    # , which huggingface_hub 2.x rejects,
    # so every task defining a vllm judge (e.g. mt_bench) failed to load.
    judge = JudgeLLM(
        judge_model_name="some-org/some-judge-model",
        template=lambda **kwargs: [],
        process_judge_response=lambda response: response,
        judge_backend=judge_backend,
    )

    assert judge.judge.backend == judge_backend
    assert judge.judge.model == "some-org/some-judge-model"


def test_mt_bench_task_module_imports():
    # mt_bench builds a vllm-backed JudgeLLM at import time.
    import lighteval.tasks.tasks.mt_bench.main  # noqa: F401
