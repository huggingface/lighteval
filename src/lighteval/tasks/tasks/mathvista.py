"""
name:
MathVista

dataset:
AI4Math/MathVista

abstract:
Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive problem-solving skills in many tasks and domains, but their ability in mathematical reasoning in visual contexts has not been systematically studied. To bridge this gap, we present MathVista, a benchmark designed to combine challenges from diverse mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets involving mathematics and 3 newly created datasets (i.e., IQTest, FunctionQA, and PaperQA). Completing these tasks requires fine-grained, deep visual understanding and compositional reasoning, which all state-of-the-art foundation models find challenging.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/pdf/2310.02255
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def multichoice_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["query"],
        choices=line["choices"],
        gold_index=line["choices"].index(line["answer"]),
        images=[line["decoded_image"]],
    )


def freeform_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["query"],
        choices=[],
        gold_index=[],
        images=[line["decoded_image"]],
    )


mathvista_freeform = LightevalTaskConfig(
    name="mathvista:freeform",
    prompt_function=freeform_prompt,
    hf_repo="AI4Math/MathVista",
    hf_subset="default",
    hf_filter=lambda x: x.get("question_type") == "free_form",
    hf_avail_splits=["testmini, test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=["\n"],
    version=0,
)

mathvista_multichoice = LightevalTaskConfig(
    name="mathvista:multichoice",
    prompt_function=multichoice_prompt,
    hf_repo="AI4Math/MathVista",
    hf_subset="default",
    hf_filter=lambda x: x.get("question_type") == "multi_choice",
    hf_avail_splits=["testmini, test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=None,
    version=0,
)

TASKS_TABLE = [
    mathvista_freeform,
    mathvista_multichoice,
]
