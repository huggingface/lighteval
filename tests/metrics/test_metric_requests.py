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

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbPMINorm
from lighteval.metrics.utils.metric_utils import Metric
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse
from lighteval.tasks.default_tasks import xstory_cloze_en_lighteval
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.requests import Doc
from tests.utils import FakeModel, fake_evaluate_task


# Doesn't matter as won't be used
def dummy_prompt_fc(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["input_sentence_1"],
        unconditioned_query="",
        gold_index=0,
        choices=["Hello", "World"],
    )


def get_pmi_task(metrics: list[Metric]):
    return LightevalTaskConfig(
        name="pmi_test_task",
        metric=metrics,
        prompt_function=dummy_prompt_fc,
        hf_repo=xstory_cloze_en_lighteval.hf_repo,
        hf_subset=xstory_cloze_en_lighteval.hf_subset,
        evaluation_splits=xstory_cloze_en_lighteval.evaluation_splits,
    )


def test_pmi_request():
    """
    Test that the PMI requests are correctly routed and computed
    """
    fake_model = FakeModel(
        loglikelihood_responses=[
            LoglikelihoodResponse(
                result=(0.9, True),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            LoglikelihoodResponse(
                result=(0.2, False),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            # Normalization loglikehioods
            LoglikelihoodResponse(
                result=(0.85, True),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            LoglikelihoodResponse(
                result=(0.1, False),
                generated_tokens=[0],
                input_tokens=[0],
            ),
        ]
    )

    metric = loglikelihood_acc_metric(normalization=LogProbPMINorm())
    pmi_test_config = get_pmi_task(metrics=[metric])
    pmi_test_config.metric = (metric,)
    task = LightevalTask(pmi_test_config.name, pmi_test_config)
    result = fake_evaluate_task(task, fake_model, max_samples=1)
    # Correct choice after norm should be the second one so 0 acc
    assert result[metric.metric_name][0] == 0


def test_pmi_request_with_logprob_metric():
    """
    Test that the PMI requests are correctly routed and computed, this ensures
    that metrics categories producing same requests are handled correctly
    """
    fake_model = FakeModel(
        loglikelihood_responses=[
            LoglikelihoodResponse(
                result=(0.9, True),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            LoglikelihoodResponse(
                result=(0.2, False),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            # Normalization loglikehioods
            LoglikelihoodResponse(
                result=(0.85, True),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            LoglikelihoodResponse(
                result=(0.1, False),
                generated_tokens=[0],
                input_tokens=[0],
            ),
        ]
    )

    metrics = [loglikelihood_acc_metric(normalization=LogProbPMINorm()), loglikelihood_acc_metric(normalization=None)]
    pmi_test_config = get_pmi_task(metrics=metrics)
    task = LightevalTask(pmi_test_config.name, pmi_test_config)
    result = fake_evaluate_task(task, fake_model, max_samples=1)
    # Correct choice after norm should be the second one so 0 acc
    assert result[metrics[0].metric_name][0] == 0
    assert result[metrics[1].metric_name][0] == 1


def test_pmi_request_with_generative_metric():
    """
    Test that the PMI requests are correctly routed even with other metrics to compute
    This is mostly that results are mutated in place, which can quickly backfire if we don't
    do it in correct order (this got actually fixed in the past but we'll keep the test for now)
    """
    fake_model = FakeModel(
        loglikelihood_responses=[
            LoglikelihoodResponse(
                result=(0.9, True),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            LoglikelihoodResponse(
                result=(0.2, False),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            # Normalization loglikehioods
            LoglikelihoodResponse(
                result=(0.85, True),
                generated_tokens=[0],
                input_tokens=[0],
            ),
            LoglikelihoodResponse(
                result=(0.1, False),
                generated_tokens=[0],
                input_tokens=[0],
            ),
        ],
        greedy_until_responses=[
            GenerativeResponse(
                result="Hello",
                generated_tokens=[0],
                input_tokens=[0],
            )
        ],
    )

    metrics = [loglikelihood_acc_metric(normalization=LogProbPMINorm()), Metrics.exact_match.value]
    pmi_test_config = get_pmi_task(metrics=metrics)
    task = LightevalTask(pmi_test_config.name, pmi_test_config)
    results = fake_evaluate_task(task, fake_model, max_samples=1)
    assert results[metrics[0].metric_name][0] == 0
    assert results[metrics[1].metric_name][0] == 1
