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

import pytest

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import Doc


@pytest.mark.parametrize("fewshot_select", ["sequential", "random", "balanced"])
def test_fewshot_sampler(fewshot_select: str):
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select=fewshot_select,
    )
    task = LightevalTask(config)
    rnd = random.Random(0)
    task._fewshot_docs = [
        Doc(str(i), ["A", "B"], rnd.randint(0, 2), fewshot_sorting_class=str(i % 20)) for i in range(100)
    ]
    sampler = FewShotSampler(task)
    seed = 1
    docs = sampler.sample_fewshot_examples(20, seed)

    match task.fewshot_selection:
        case "sequential":
            assert docs == task.fewshot_docs()[20:40]
        case "random":
            rnd = random.Random(seed)
            task_docs = task.fewshot_docs()
            rnd.shuffle(task_docs)
            assert docs == task_docs[:20]


def test_sequential_fewshot_sampling_keeps_each_seed_independent():
    config = LightevalTaskConfig(
        name="test_sequential_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select="sequential",
    )
    task = LightevalTask(config)
    task._fewshot_docs = [Doc(str(i), ["A", "B"], 0) for i in range(10)]
    sampler = FewShotSampler(task)

    sampled_by_seed = {seed: [doc.query for doc in sampler.sample_fewshot_examples(2, seed)] for seed in (0, 1, 2)}

    assert sampled_by_seed == {0: ["0", "1"], 1: ["2", "3"], 2: ["4", "5"]}
    assert [doc.query for doc in task.fewshot_docs()] == [str(i) for i in range(10)]


def test_balanced_fewshot_sampling_accepts_falsy_labels():
    config = LightevalTaskConfig(
        name="test_balanced_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select="balanced",
    )
    task = LightevalTask(config)
    task._fewshot_docs = [Doc(f"empty-{i}", ["", "x"], 0) for i in range(10)] + [
        Doc(f"value-{i}", ["", "x"], 1) for i in range(10)
    ]

    sampled = FewShotSampler(task).sample_fewshot_examples(4, variance_seed=0)

    assert len(sampled) == 4


def test_balanced_fewshot_sampling_does_not_mutate_global_random_state():
    config = LightevalTaskConfig(
        name="test_balanced_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select="balanced",
    )
    task = LightevalTask(config)
    task._fewshot_docs = [Doc(str(i), ["A", "B"], i % 2) for i in range(10)]

    random.seed(12345)
    expected_next_value = random.random()
    random.seed(12345)
    FewShotSampler(task).sample_fewshot_examples(4, variance_seed=7)

    assert random.random() == expected_next_value
