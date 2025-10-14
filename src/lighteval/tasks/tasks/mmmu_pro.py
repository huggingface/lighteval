"""
MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the
true understanding capabilities of advanced AI models across multiple
modalities.

languages:
en

tags:
general-knowledge, knowledge, multimodal, multiple-choice

paper:
https://arxiv.org/abs/2409.02813
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


mmmu_pro_standard_4_options = LightevalTaskConfig(
    name="mmmu_pro:standard-4",
    suite=["lighteval"],
    prompt_function=prompt.mmmu_pro,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="standard (4 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


mmmu_pro_standard_10_options = LightevalTaskConfig(
    name="mmmu_pro:standard-10",
    suite=["lighteval"],
    prompt_function=prompt.mmmu_pro,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="standard (10 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


mmmu_pro_vision = LightevalTaskConfig(
    name="mmmu_pro:vision",
    suite=["lighteval"],
    prompt_function=prompt.mmmu_pro_vision,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="vision",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)
