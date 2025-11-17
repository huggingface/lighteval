"""
name:
Gpqa

dataset:
Idavidrein/gpqa

abstract:
GPQA is a dataset of 448 expert-written multiple-choice questions in biology,
physics, and chemistry, designed to test graduate-level reasoning. The questions
are extremely difficult—PhD-level experts score about 65%, skilled non-experts
34% (even with web access), and GPT-4 around 39%. GPQA aims to support research
on scalable oversight, helping humans evaluate and trust AI systems that may
exceed human expertise.

languages:
english

tags:
biology, chemistry, graduate-level, multiple-choice, physics, qa, reasoning, science

paper:
https://arxiv.org/abs/2311.12022
"""

import random
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def record_to_sample(record):
    gold_index = random.randint(0, 3)
    choices = [record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
    choices.insert(gold_index, record["Correct Answer"])
    return Sample(
        input=record["Question"].strip(),
        choices=choices,
        target=ascii_uppercase[gold_index],
    )


def sample_to_fewshot(sample):
    return f"{sample.input}\n\n" + f"ANSWER: {sample.target}"


gpqa = LightevalTaskConfig(
    name="gpqa:mc",
    prompt_function=prompt.gpqa,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

gpqa_diamond_instruct = LightevalTaskConfig(
    name="gpqa:diamond",
    prompt_function=prompt.gpqa_instruct,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1})],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=1,
)

gpqa_extended_instruct = LightevalTaskConfig(
    name="gpqa:extended",
    prompt_function=prompt.gpqa_instruct,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_extended",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)

gpqa_main_instruct = LightevalTaskConfig(
    name="gpqa:main",
    prompt_function=prompt.gpqa_instruct,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)

TASKS_TABLE = [
    gpqa,
    gpqa_diamond_instruct,
    gpqa_extended_instruct,
    gpqa_main_instruct,
]
