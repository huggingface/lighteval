"""
name:
OZ Serbian Evals

dataset:
DjMel/oz-eval

abstract:
OZ Eval (sr. Opšte Znanje Evaluacija) dataset was created for the purposes of
evaluating General Knowledge of LLM models in Serbian language.  Data consists
of 1k+ high-quality questions and answers which were used as part of entry exams
at the Faculty of Philosophy and Faculty of Organizational Sciences, University
of Belgrade.  The exams test the General Knowledge of students and were used in
the enrollment periods from 2003 to 2024.

languages:
serbian

tags:
knowledge, multiple-choice

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_fn_oz_eval_task(line, task_name: str = None):
    query_template = """Pitanje: {question}\n
    Ponuđeni odgovori:
    A. {choice_a}
    B. {choice_b}
    C. {choice_c}
    D. {choice_d}
    E. {choice_e}

    Krajnji odgovor:"""

    options = line["options"]

    query = query_template.format(
        question=line["questions"],
        choice_a=options[0],
        choice_b=options[1],
        choice_c=options[2],
        choice_d=options[3],
        choice_e=options[4],
    )

    choices = ["A", "B", "C", "D", "E"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["answer"]),
    )


oz_eval_task = LightevalTaskConfig(
    name="serbian_evals:oz_task",
    prompt_function=prompt_fn_oz_eval_task,
    hf_repo="DjMel/oz-eval",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.loglikelihood_acc],
    version=0,
)


# STORE YOUR EVALS
TASKS_TABLE = [oz_eval_task]
