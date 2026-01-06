"""
name:
Mix Eval

dataset:
MixEval/MixEval

abstract:
Ground-truth-based dynamic benchmark derived from off-the-shelf benchmark
mixtures, which evaluates LLMs with a highly capable model ranking (i.e., 0.96
correlation with Chatbot Arena) while running locally and quickly (6% the time
and cost of running MMLU), with its queries being stably and effortlessly
updated every month to avoid contamination.

languages:
english

tags:
general-knowledge, reasoning, qa

paper:
https://mixeval.github.io/

starred:
true
"""

import logging
import re
from string import ascii_uppercase

import numpy as np
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice, model_graded_fact
from inspect_ai.solver import generate, multiple_choice

from lighteval.metrics.metrics_sample import JudgeLLMMixEval
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.mix_eval.judge_prompts import (
    flow_judge_for_freeform_template,
    flow_judge_for_multichoice_template,
    gpt_judge_for_closeended_freeform,
    gpt_judge_for_closeended_multiplechoice,
)
from lighteval.tasks.tasks.mix_eval.prompts import construct_prompt_freeform, construct_prompt_multichoice


logger = logging.getLogger(__name__)


def mixeval_freeform_prompt(line, task_name: str = ""):
    prompt = construct_prompt_freeform(line)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=line["target"],
        gold_index=list(range(len(line["target"]))),
        instruction="",
        specific={
            "problem-type": line["problem_type"],
            "benchmark-name": line["benchmark_name"],
            "question": line["prompt"],
        },
    )


# Very specific task where there are no precise outputs but instead we test if the format obeys rules
def mixeval_multichoice_prompt(line, task_name: str = ""):
    prompt = construct_prompt_multichoice(line)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=line["options"],
        gold_index=[int(target) for target in line["target"]],
        instruction="",
        specific={
            "problem-type": line["problem_type"],
            "benchmark-name": line["benchmark_name"],
            "question": line["prompt"],
        },
    )


def record_to_sample_freeform(record):
    query = record["prompt"]
    target = record["target"][0]
    return Sample(input=query, target=target)


def record_to_sample_multichoice(record):
    query = record["prompt"]
    choices = record["options"]
    target = ascii_uppercase[int(record["target"][0])]
    return Sample(input=query, target=target, choices=choices)


def process_judge_response(x):
    try:
        search = re.search(r"<score>\s(\d)\s</score>", x)
        return int(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for flow: {e}")
        return 0


def process_judge_response_multichoice_gpt(x):
    try:
        search = re.search(r"\[\[([01])\]\]", x)
        return int(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for multichoice GPT: {e}")
        return 0


def process_judge_response_freeform_gpt(x):
    try:
        search = re.search(r"\[\[(\d.\d)\]\]", x)
        return float(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for freeform GPT: {e}")
        return 0


llm_judge_mixeval_multichoice_flow_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_flow"],
    higher_is_better={"judge_score_flow": True},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_for_multichoice_template,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
        short_judge_name="flow",
    ),
    corpus_level_fn={
        "judge_score_flow": np.mean,
    },
    batched_compute=True,
)

llm_judge_mixeval_multichoice_gpt_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_gpt3"],
    higher_is_better={"judge_score_gpt-3.5": True},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="gpt-3.5-turbo",
        template=gpt_judge_for_closeended_multiplechoice,
        process_judge_response=process_judge_response_multichoice_gpt,
        judge_backend="openai",
        short_judge_name="gpt-3.5",
    ),
    corpus_level_fn={
        "judge_score_gpt-3.5": np.mean,
    },
    batched_compute=True,
)


def mean_dv_5(x):
    return np.mean(x) / 5


llm_judge_mixeval_freeform_flow_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_flow"],
    higher_is_better={"judge_score": True},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_for_freeform_template,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
        short_judge_name="flow",
    ),
    corpus_level_fn={
        "judge_score_flow": mean_dv_5,
    },
    batched_compute=True,
)

llm_judge_mixeval_freeform_gpt_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_gpt3"],
    higher_is_better={"judge_score_gpt-3.5": True},
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="gpt-3.5-turbo",
        template=gpt_judge_for_closeended_freeform,
        process_judge_response=process_judge_response_freeform_gpt,
        judge_backend="openai",
        short_judge_name="gpt-3.5",
    ),
    corpus_level_fn={
        "judge_score_gpt-3.5": np.mean,
    },
    batched_compute=True,
)


mixeval_freeform_easy = LightevalTaskConfig(
    name="mixeval_easy:freeform",
    prompt_function=mixeval_freeform_prompt,
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval",
    metrics=[llm_judge_mixeval_freeform_flow_judge, llm_judge_mixeval_freeform_gpt_judge],
    hf_avail_splits=["free_form"],
    evaluation_splits=["free_form"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
    sample_fields=record_to_sample_freeform,
    solver=[generate(cache=True)],
    scorer=model_graded_fact(),
)


mixeval_multichoice_easy = LightevalTaskConfig(
    name="mixeval_easy:multichoice",
    prompt_function=mixeval_multichoice_prompt,
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval",
    metrics=[llm_judge_mixeval_multichoice_flow_judge, llm_judge_mixeval_multichoice_gpt_judge],
    hf_avail_splits=["multiple_choice"],
    evaluation_splits=["multiple_choice"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
    sample_fields=record_to_sample_multichoice,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

mixeval_freeform_hard = LightevalTaskConfig(
    name="mixeval_hard:freeform",
    prompt_function=mixeval_freeform_prompt,
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval_Hard",
    metrics=[llm_judge_mixeval_freeform_flow_judge, llm_judge_mixeval_freeform_gpt_judge],
    hf_avail_splits=["free_form"],
    evaluation_splits=["free_form"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
    sample_fields=record_to_sample_freeform,
    solver=[generate(cache=True)],
    scorer=model_graded_fact(),
)


mixeval_multichoice_hard = LightevalTaskConfig(
    name="mixeval_hard:multichoice",
    prompt_function=mixeval_multichoice_prompt,
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval_Hard",
    metrics=[llm_judge_mixeval_multichoice_flow_judge, llm_judge_mixeval_multichoice_gpt_judge],
    hf_avail_splits=["multiple_choice"],
    evaluation_splits=["multiple_choice"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
    sample_fields=record_to_sample_multichoice,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


TASKS_TABLE = [mixeval_multichoice_easy, mixeval_freeform_easy, mixeval_multichoice_hard, mixeval_freeform_hard]
