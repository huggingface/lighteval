"""
name:
Triviaqa

dataset:
mandarjoshi/trivia_qa

abstract:
TriviaqQA is a reading comprehension dataset containing over 650K
question-answer-evidence triples. TriviaqQA includes 95K question-answer pairs
authored by trivia enthusiasts and independently gathered evidence documents,
six per question on average, that provide high quality distant supervision for
answering the questions.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1705.03551
"""

import string

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def triviaqa_prompt(line, task_name: str = None):
    def _remove_prefixes(aliases):
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)
        return ret

    list_of_candidates = [
        alias.lower().translate(str.maketrans("", "", string.punctuation))
        for alias in _remove_prefixes(line["answer"]["aliases"])
    ]

    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        gold_index=0,
        choices=[list_of_candidates],
    )


triviaqa = LightevalTaskConfig(
    name="triviaqa",
    prompt_function=triviaqa_prompt,
    hf_repo="mandarjoshi/trivia_qa",
    hf_subset="rc.nocontext",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", ".", ","],
    version=0,
)

TASKS_TABLE = [
    triviaqa,
]
