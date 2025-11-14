"""
name:
Narrativeqa

dataset:
lighteval/narrative_qa_helm

abstract:
NarrativeQA is a reading comprehension benchmark that tests deep understanding
of full narratives—books and movie scripts—rather than shallow text matching. To
answer its questions, models must integrate information across entire stories.

languages:
english

tags:
qa, reading-comprehension

paper:
https://aclanthology.org/Q18-1023/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


narrativeqa_instruction = "Answer the question based on the passage.\n"


def narrativeqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {line['question']}\nAnswer:",
        gold_index=list(range(len(line["references"]))),
        choices=[[str(a) for a in line["references"]]],
    )


narrativeqa = LightevalTaskConfig(
    name="narrativeqa",
    prompt_function=narrativeqa_prompt,
    hf_repo="lighteval/narrative_qa_helm",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    narrativeqa,
]
