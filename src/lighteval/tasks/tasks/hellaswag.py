"""
name:
Hellaswag

dataset:
Rowan/hellaswag

abstract:
HellaSwag is a commonsense inference benchmark designed to challenge language
models with adversarially filtered multiple-choice questions.

languages:
english

tags:
multiple-choice, narrative, reasoning

paper:
https://arxiv.org/abs/1905.07830
"""

from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.hellaswag import hellaswag_preprocess


def hellaswag_prompt(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n\n"
    query += f"Question: {line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["endings"])])
    query += "Answer:"

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in ascii_uppercase[: len(line["endings"])]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n\n",
    )


def hellaswag_harness_prompt(line, task_name: str = None):
    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=hellaswag_preprocess(f"{line['activity_label']}: {ctx}"),
        choices=[hellaswag_preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,
    )


hellaswag = LightevalTaskConfig(
    name="hellaswag",
    prompt_function=hellaswag_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

hellaswag_harness = LightevalTaskConfig(
    name="hellaswag_harness",
    prompt_function=hellaswag_harness_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=[
        LogLikelihoodAccMetric(),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    hellaswag,
    hellaswag_harness,
]
