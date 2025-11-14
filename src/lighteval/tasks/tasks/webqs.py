"""
name:
Webqs

dataset:
stanfordnlp/web_questions

abstract:
This dataset consists of 6,642 question/answer pairs. The questions are supposed
to be answerable by Freebase, a large knowledge graph. The questions are mostly
centered around a single named entity. The questions are popular ones asked on
the web.

languages:
english

tags:
qa

paper:
https://aclanthology.org/D13-1160.pdf
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def webqs_prompt(line, task_name: str = None):
    def _remove_prefixes(aliases):
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)
        return ret

    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        gold_index=0,
        choices=[[f" {c}" for c in _remove_prefixes(line["answers"])]],
    )


webqs = LightevalTaskConfig(
    name="webqs",
    prompt_function=webqs_prompt,
    hf_repo="stanfordnlp/web_questions",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    webqs,
]
