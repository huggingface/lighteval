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
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language


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

hellaswag_cf = LightevalTaskConfig(
    name="hellaswag_cf",
    prompt_function=get_hellaswag_prompt_function(
        language=Language.ENGLISH,
        adapter=lambda line: {
            "activity_label": line["activity_label"],
            "ctx_a": line["ctx_a"],
            "ctx_b": line["ctx_b"],
            "continuations": line["endings"],
            "gold_idx": int(line["label"]) if line["label"] != "" else -1,
        },
        formulation=CFFormulation(),
    ),
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[
        LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
        LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    ],
    version=0,
)

TASKS_TABLE = [
    hellaswag,
    hellaswag_cf,
]
