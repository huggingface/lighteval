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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


"""
abstract:
AGIEval is a human-centric benchmark specifically designed to evaluate the
general abilities of foundation models in tasks pertinent to human cognition and
problem-solving. This benchmark is derived from 20 official, public, and
high-standard admission and qualification exams intended for general human
test-takers, such as general college admission tests (e.g., Chinese College
Entrance Exam (Gaokao) and American SAT), law school admission tests, math
competitions, lawyer qualification tests, and national civil service exams.

languages:
en, zh

tags:
math, reasoning, law, language, history, chemistry, biology, geography, physics

paper:
https://arxiv.org/abs/2304.06364
"""

agieval_aqua_rat = LightevalTaskConfig(
    name="agieval:aqua-rat",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-aqua-rat",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_biology = LightevalTaskConfig(
    name="agieval:gaokao-biology",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-biology",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_chemistry = LightevalTaskConfig(
    name="agieval:gaokao-chemistry",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-chemistry",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_chinese = LightevalTaskConfig(
    name="agieval:gaokao-chinese",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-chinese",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_english = LightevalTaskConfig(
    name="agieval:gaokao-english",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-english",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_geography = LightevalTaskConfig(
    name="agieval:gaokao-geography",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-geography",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_history = LightevalTaskConfig(
    name="agieval:gaokao-history",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-history",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_mathqa = LightevalTaskConfig(
    name="agieval:gaokao-mathqa",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-mathqa",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_gaokao_physics = LightevalTaskConfig(
    name="agieval:gaokao-physics",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-gaokao-physics",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_logiqa_en = LightevalTaskConfig(
    name="agieval:logiqa-en",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-logiqa-en",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_logiqa_zh = LightevalTaskConfig(
    name="agieval:logiqa-zh",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-logiqa-zh",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_lsat_ar = LightevalTaskConfig(
    name="agieval:lsat-ar",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-lsat-ar",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_lsat_lr = LightevalTaskConfig(
    name="agieval:lsat-lr",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-lsat-lr",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_lsat_rc = LightevalTaskConfig(
    name="agieval:lsat-rc",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-lsat-rc",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_sat_en = LightevalTaskConfig(
    name="agieval:sat-en",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-sat-en",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_sat_en_without_passage = LightevalTaskConfig(
    name="agieval:sat-en-without-passage",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-sat-en-without-passage",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)

agieval_sat_math = LightevalTaskConfig(
    name="agieval:sat-math",
    suite=["lighteval"],
    prompt_function=prompt.agieval,
    hf_repo="dmayhem93/agieval-sat-math",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=None,
    version=0,
)
