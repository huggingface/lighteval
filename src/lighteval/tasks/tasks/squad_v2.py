"""
name:
Squad V2

dataset:
rajpurkar/squad_v2

abstract:
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
consisting of questions posed by crowdworkers on a set of Wikipedia articles,
where the answer to every question is a segment of text, or span, from the
corresponding reading passage, or the question might be unanswerable.
SQuAD 2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000
unanswerable questions written adversarially by crowdworkers to look similar to
answerable ones. To do well on SQuAD2.0, systems must not only answer questions
when possible, but also determine when no answer is supported by the paragraph
and abstain from answering.

note:
This is an LLM-friendly adaptation of the original SQuAD 2.0 evaluation.
The original evaluation uses extractive span selection with a confidence-based
"no answer" threshold, which does not apply to generative models.
Here, the model is instead instructed to generate "unanswerable" when the
question cannot be answered from the context. EM and F1 metrics are computed
over both answerable and unanswerable questions.

languages:
english

tags:
qa

paper:
https://arxiv.org/abs/1806.03822
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language

UNANSWERABLE = "unanswerable"


def squad_v2_prompt(line, task_name: str = None):
    answers = list(set(ans for ans in line["answers"]["text"] if len(ans) > 0))
    is_unanswerable = len(answers) == 0

    if is_unanswerable:
        choices = [f" {UNANSWERABLE}"]
    else:
        choices = [f" {ans}" for ans in answers]

    return Doc(
        task_name=task_name,
        query=f"Context: {line['context']}\nQuestion: {line['question']}\n"
        f"Answer with a span from the context, or \"{UNANSWERABLE}\" if the question cannot be answered.\nAnswer:",
        choices=choices,
        gold_index=list(range(len(choices))),
    )


squad_v2 = LightevalTaskConfig(
    name="squad_v2",
    prompt_function=squad_v2_prompt,
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    evaluation_splits=("validation",),
    few_shots_split="train",
    stop_sequence=["\n", "Question:", "question:"],
    generation_size=200,
    metrics=[Metrics.exact_match, Metrics.f1_score],
    version=2,
)

squad_v2_answerable = LightevalTaskConfig(
    name="squad_v2:answerable",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "context": line["context"],
            "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
        },
    ),
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    hf_filter=lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    evaluation_splits=("validation",),
    few_shots_split="train",
    stop_sequence=["\n", "Question:", "question:"],
    generation_size=200,
    metrics=[Metrics.exact_match, Metrics.f1_score],
    version=1,
)

TASKS_TABLE = [
    squad_v2,
    squad_v2_answerable,
]
