"""
abstract:
ASDiv is a dataset for arithmetic reasoning that contains 2,000+ questions
covering addition, subtraction, multiplication, and division.

languages:
en

paper:
https://arxiv.org/abs/2410.12853

tags:
math, reasoning
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


asdiv = LightevalTaskConfig(
    name="asdiv",
    suite=["lighteval"],
    prompt_function=prompt.asdiv,
    hf_repo="EleutherAI/asdiv",
    hf_subset="asdiv",
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)
