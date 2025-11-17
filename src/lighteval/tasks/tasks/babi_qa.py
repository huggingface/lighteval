"""
name:
Babi Qa

dataset:
facebook/babi_qa

abstract:
The bAbI benchmark for measuring understanding and reasoning, evaluates reading
comprehension via question answering.

languages:
english

tags:
qa, reasoning

paper:
https://arxiv.org/abs/1502.05698
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


babi_qa = LightevalTaskConfig(
    name="babi_qa",
    prompt_function=prompt.babi_qa,
    hf_repo="facebook/babi_qa",
    hf_subset="en-valid-qa1",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [babi_qa]
