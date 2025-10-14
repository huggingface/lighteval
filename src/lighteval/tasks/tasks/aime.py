"""
abstract:
The American Invitational Mathematics Examination (AIME) is a prestigious,
invite-only mathematics competition for high-school students who perform in the
top 5% of the AMC 12 mathematics exam. It involves 15 questions of increasing
difficulty, with the answer to every question being a single integer from 0 to
999. The median score is historically between 4 and 6 questions correct (out of
the 15 possible). Two versions of the test are given every year (thirty
questions total).

languages:
en

paper:
https://maa.org/aime-thresholds-are-available/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1})],
    version=2,
)

aime24_gpassk = LightevalTaskConfig(
    name="aime24_gpassk",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10000,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1})],
    version=2,
)

aime25_gpassk = LightevalTaskConfig(
    name="aime25_gpassk",
    suite=["lighteval"],
    prompt_function=prompt.aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)
