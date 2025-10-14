"""
abstract:
The SciQ dataset contains 13,679 crowdsourced science exam questions about
Physics, Chemistry and Biology, among others. The questions are in
multiple-choice format with 4 answer options each. For the majority of the
questions, an additional paragraph with supporting evidence for the correct
answer is provided.

languages:
en

tags:
physics, chemistry, biology, reasoning, multiple-choice, qa

paper:
https://arxiv.org/abs/1707.06209
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


sciq = LightevalTaskConfig(
    name="sciq",
    suite=["lighteval"],
    prompt_function=prompt.sciq,
    hf_repo="sciq",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)
