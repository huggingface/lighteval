"""
name:
Sciq

dataset:
allenai/sciq

abstract:
The SciQ dataset contains 13,679 crowdsourced science exam questions about
Physics, Chemistry and Biology, among others. The questions are in
multiple-choice format with 4 answer options each. For the majority of the
questions, an additional paragraph with supporting evidence for the correct
answer is provided.

languages:
english

tags:
physics, chemistry, biology, reasoning, multiple-choice, qa

paper:
https://arxiv.org/abs/1707.06209
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def sciq_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['support']}\nQuestion: {line['question']}\nAnswer:".strip(),
        choices=[
            f" {c}" for c in [line["distractor1"], line["distractor2"], line["distractor3"], line["correct_answer"]]
        ],
        gold_index=3,
    )


sciq = LightevalTaskConfig(
    name="sciq",
    prompt_function=sciq_prompt,
    hf_repo="allenai/sciq",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    sciq,
]
