"""
abstract:
LogiQA is a machine reading comprehension dataset focused on testing logical
reasoning abilities. It contains 8,678 expert-written multiple-choice questions
covering various types of deductive reasoning. While humans perform strongly,
state-of-the-art models lag far behind, making LogiQA a benchmark for advancing
logical reasoning in NLP systems.

languages:
en

tags:
reading-comprehension

paper:
https://arxiv.org/abs/2007.08124
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


logiqa = LightevalTaskConfig(
    name="logiqa",
    suite=["lighteval"],
    prompt_function=prompt.logiqa,
    hf_repo="lighteval/logiqa_harness",
    hf_subset="logiqa",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)
