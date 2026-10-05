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

import pytest

from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks.hellaswag import (
    TASKS_TABLE as HELLASWAG_TASKS_TABLE,
)
from lighteval.tasks.tasks.hellaswag import (
    hellaswag,
    hellaswag_harness,
    hellaswag_harness_prompt,
)
from lighteval.tasks.tasks.piqa import (
    TASKS_TABLE as PIQA_TASKS_TABLE,
)
from lighteval.tasks.tasks.piqa import (
    piqa,
    piqa_harness,
    piqa_harness_prompt,
)


def test_hellaswag_harness_prompt():
    line = {
        "activity_label": "Removing ice",
        "ctx_a": "A person grabs a pick.",
        "ctx_b": "they chip at the ice",
        "endings": ["The ice breaks [title] away.", "The ice grows."],
        "label": "0",
    }

    doc = hellaswag_harness_prompt(line, "hellaswag_harness")

    assert doc.query == "Removing ice: A person grabs a pick. They chip at the ice "
    assert doc.choices == ["The ice breaks. away.", "The ice grows."]
    assert doc.gold_index == 0


def test_piqa_harness_prompt():
    line = {
        "goal": "Keep a door open",
        "sol1": "Use a doorstop",
        "sol2": "Lock the door",
        "label": 0,
    }

    doc = piqa_harness_prompt(line, "piqa_harness")

    assert doc.query == "Question: Keep a door open\nAnswer:"
    assert doc.choices == [" Use a doorstop", " Lock the door"]
    assert doc.gold_index == 0


@pytest.mark.parametrize(
    ("config", "ignore_first_space"),
    [
        (hellaswag_harness, False),
        (piqa_harness, True),
    ],
)
def test_harness_tasks_use_loglikelihood(config, ignore_first_space):
    task = LightevalTask(config)

    assert task.sampling_methods == [SamplingMethod.LOGPROBS]
    assert task.generation_size == -1
    assert [metric.metric_name for metric in task.metrics] == ["acc", "acc_norm"]

    normalization = task.metrics[1].sample_level_fn.logprob_normalization
    assert isinstance(normalization, LogProbCharNorm)
    assert normalization.ignore_first_space is ignore_first_space


@pytest.mark.parametrize("config", [hellaswag, piqa])
def test_existing_tasks_remain_generative(config):
    task = LightevalTask(config)

    assert task.sampling_methods == [SamplingMethod.GENERATIVE]


def test_harness_tasks_are_exported():
    assert {config.name for config in HELLASWAG_TASKS_TABLE} == {"hellaswag", "hellaswag_harness"}
    assert {config.name for config in PIQA_TASKS_TABLE} == {"piqa", "piqa_harness"}
