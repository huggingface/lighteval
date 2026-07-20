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
    original_docs = list(task._fewshot_docs)
    sampler = FewShotSampler(task)
    seed = 1
    docs = sampler.sample_fewshot_examples(20, seed)

    match task.fewshot_selection:
        case "sequential":
            # Sequential selection rotates the pool by `num_fewshot * seed` from the original order.
            rot = (20 * seed) % len(original_docs)
            assert docs == (original_docs[rot:] + original_docs[:rot])[:20]
        case "random":
            rnd = random.Random(seed)
            task_docs = list(original_docs)
            rnd.shuffle(task_docs)
            assert docs == task_docs[:20]


def _make_sequential_task(num_docs: int = 100) -> LightevalTask:
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select="sequential",
    )
    task = LightevalTask(config)
    task._fewshot_docs = [Doc(str(i), ["A", "B"], 0) for i in range(num_docs)]
    return task


def test_sequential_fewshot_does_not_mutate_shared_pool():
    """`fewshot_docs()` returns the task's memoized pool by reference. The sequential sampler rotates
    it to offset the selection per seed, and must not do so in place, otherwise the shared pool is
    corrupted for every subsequent seed (variance evaluation reuses the same pool across seeds)."""
    task = _make_sequential_task()
    original_order = list(task._fewshot_docs)

    sampler = FewShotSampler(task)
    # A single non-zero seed is enough: it rotates the pool by `num_fewshot * seed`.
    sampler.sample_fewshot_examples(num_fewshot=20, variance_seed=3)

    assert task.fewshot_docs() == original_order


def test_sequential_fewshot_seeds_are_independent():
    """Each variance seed must select the pool rotated by `num_fewshot * seed` from the *original*
    order. Rotating the shared pool in place makes the rotations accumulate across seeds (and aliases
    every seed's cached selection to the same list), so later seeds get the wrong examples."""
    task = _make_sequential_task()
    original_order = list(task._fewshot_docs)
    num_fewshot = 20

    sampler = FewShotSampler(task)
    results = {seed: sampler.sample_fewshot_examples(num_fewshot=num_fewshot, variance_seed=seed) for seed in (0, 1, 2)}

    for seed in (0, 1, 2):
        rot = (num_fewshot * seed) % len(original_order)
        expected = (original_order[rot:] + original_order[:rot])[:num_fewshot]
        assert results[seed] == expected, f"seed {seed} selected the wrong few-shot examples"
