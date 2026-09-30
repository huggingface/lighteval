"""
name:
Pubmedqa

dataset:
pubmed_qa

abstract:
PubMedQA is a dataset for biomedical research question answering.

languages:
english

tags:
biomedical, health, medical, qa

paper:
https://pubmedqa.github.io/
"""

import numpy as np

from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def pubmed_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['question']}\n{line['context']['contexts']}\nAnswer: ",
        choices=[line["final_decision"]],
        gold_index=0,
    )


exact_match_case_insensitive = SampleLevelMetric(
    metric_name="em_ci",
    sample_level_fn=ExactMatches(strip_strings=True, normalize_gold=str.lower, normalize_pred=str.lower),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

pubmedqa = LightevalTaskConfig(
    name="pubmedqa",
    prompt_function=pubmed_qa_prompt,
    hf_repo="pubmed_qa",
    hf_subset="pqa_labeled",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        exact_match_case_insensitive,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    pubmedqa,
]
