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
from collections import Counter

import pytest

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import FewShotSampler, PromptManager
from lighteval.tasks.requests import Doc


@pytest.mark.parametrize("fewshot_select", ["sequential", "random", "balanced"])
def test_fewshot_sampler(fewshot_select: str):
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo=None,
        hf_subset="default",
        metric=[],
        few_shots_split="test",
        few_shots_select=fewshot_select,
    )
    task = LightevalTask("test_fewshot_task", config)
    rnd = random.Random(0)
    task._fewshot_docs = [
        Doc(str(i), ["A", "B"], rnd.randint(0, 2), fewshot_sorting_class=str(i % 20)) for i in range(100)
    ]
    sampler = FewShotSampler(task)
    seed = 1
    docs = sampler.sample_fewshot_examples(20, seed)
    match task.fewshot_selection:
        case "balanced":
            labels = Counter([PromptManager.doc_to_fewshot_sorting_class(d) for d in docs])
            assert labels.total() / len(labels) == 1
        case "sequential":
            assert docs == task.fewshot_docs()[:20]
        case "random":
            rnd = random.Random(seed)
            task_docs = task.fewshot_docs()
            rnd.shuffle(task_docs)
            assert docs == task_docs[:20]
