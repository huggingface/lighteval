# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.
This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
Author:
"""

from pprint import pprint

import numpy as np
from aenum import extend_enum
from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.metrics import Metrics
from lighteval.metrics.utils import MetricCategory, MetricUseCase, SampleLevelMetric, SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES
from tasks_examples.custom_tasks_with_custom_metrics.mt_bench.judges import (
    load_judge_prompts,
    make_judge_single,
    play_a_match_single,
)


NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

## EVAL WITH NO SUBSET ##
# This is how you create a simple tasks (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task = LightevalTaskConfig(
    name="mt_bench",
    prompt_function="prompt_fn",  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["custom"],
    hf_repo="HuggingFaceH4/mt_bench_prompts",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=["mt_bench_metric"],
    generation_size=100,
    stop_sequence=["."],
)


## DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks
def prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py, or get more info
    about what this function should do in the README.
    """
    return Doc(
        task_name=task_name,
        query=line["prompt"][0],
        choices=None,
        instruction="",
        gold_index=[],
        specific={"reference": line["reference"], "category": line["category"], "queries": line["prompt"]},
    )




def mt_bench_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict[str, float]:
    """Defines how to go from a list of predictions to a score.
    Follow examples in src/lighteval/metrics/metrics.py, or get more info
    about what this function should do in the README.
    """
    judge_model = "gpt-3.5-turbo"
    judge_file = "/Users/nathan/Repos/lighteval/tasks_examples/custom_tasks_with_custom_metrics/mt_bench/judge_prompts.jsonl"
    judge_prompts = load_judge_prompts(judge_file)
    judges = make_judge_single(judge_model, judge_prompts)

    question = formatted_doc.specific["queries"]
    ref_answer = formatted_doc.specific["reference"]
    category = formatted_doc.specific["category"]

    if category not in NEED_REF_CATS:
        score = play_a_match_single(question, predictions, ref_answer, judges["default"], multi_turn=False, output_file=None)
        score_mt = play_a_match_single(question, predictions, ref_answer, judges["default-mt"], multi_turn=True, output_file=None)
    else:
        try:
            score = play_a_match_single(question, predictions, ref_answer, judges["math"], multi_turn=False, output_file=None)
            score_mt = play_a_match_single(question, predictions, ref_answer, judges["math-mt"], multi_turn=True, output_file=None)
        except KeyError:
            print(f"Category {category} not found in judge prompts, using default judge")
            score = play_a_match_single(question, predictions, ref_answer, judges["default"], multi_turn=False, output_file=None)
            score_mt = play_a_match_single(question, predictions, ref_answer, judges["default-mt"], multi_turn=True, output_file=None)

    return score


mt_bench_metric = SampleLevelMetric(
    metric="mt_bench_metric",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE_MULTI_TURN,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=mt_bench_metric,
    corpus_level_fn=np.mean,
)


## STORE YOUR EVALS
_TASKS = [task]

## MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]
extend_enum(
    Metrics,
    "mt_bench_metric",
    mt_bench_metric,
)

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
