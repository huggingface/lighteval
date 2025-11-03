"""
name:
Agieval

dataset:
dmayhem93/agieval-aqua-rat, dmayhem93/agieval-gaokao-biology, dmayhem93/agieval-gaokao-chemistry, dmayhem93/agieval-gaokao-chinese, dmayhem93/agieval-gaokao-english, dmayhem93/agieval-gaokao-geography, dmayhem93/agieval-gaokao-history, dmayhem93/agieval-gaokao-mathqa, dmayhem93/agieval-gaokao-physics, dmayhem93/agieval-logiqa-en, dmayhem93/agieval-logiqa-zh, dmayhem93/agieval-lsat-ar, dmayhem93/agieval-lsat-lr, dmayhem93/agieval-lsat-rc, dmayhem93/agieval-sat-en, dmayhem93/agieval-sat-en-without-passage, dmayhem93/agieval-sat-math

abstract:
AGIEval is a human-centric benchmark specifically designed to evaluate the
general abilities of foundation models in tasks pertinent to human cognition and
problem-solving. This benchmark is derived from 20 official, public, and
high-standard admission and qualification exams intended for general human
test-takers, such as general college admission tests (e.g., Chinese College
Entrance Exam (Gaokao) and American SAT), law school admission tests, math
competitions, lawyer qualification tests, and national civil service exams.

languages:
english, chinese

tags:
biology, chemistry, geography, history, knowledge, language, multiple-choice, physics, reasoning

paper:
https://arxiv.org/abs/2304.06364
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def record_to_sample(record):
    # we need to remove prepended (A), (B), (C), (D) from the choices
    choices = [
        c.replace("(A)", "").replace("(B)", "").replace("(C)", "").replace("(D)", "").strip()
        for c in record["choices"]
    ]
    return Sample(input=record["query"], target=ascii_uppercase[record["gold"][0]], choices=choices)


agieval_aqua_rat = LightevalTaskConfig(
    name="agieval:aqua-rat",
    suite=["lighteval"],
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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

TASKS_TABLE = [
    agieval_aqua_rat,
    agieval_gaokao_biology,
    agieval_gaokao_chemistry,
    agieval_gaokao_chinese,
    agieval_gaokao_english,
    agieval_gaokao_geography,
    agieval_gaokao_history,
    agieval_gaokao_mathqa,
    agieval_gaokao_physics,
    agieval_logiqa_en,
    agieval_logiqa_zh,
    agieval_lsat_ar,
    agieval_lsat_lr,
    agieval_lsat_rc,
    agieval_sat_en,
    agieval_sat_en_without_passage,
    agieval_sat_math,
]
