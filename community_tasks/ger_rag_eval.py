# MIT License

# Copyright (c) 2024 The HuggingFace Team
# Copyright (c) 2024 Philip May, Deutsche Telekom AG

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
This module implements the 4 tasks of deutsche-telekom/Ger-RAG-eval.
See: https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


task1 = LightevalTaskConfig(
    name="ger_rag_eval_task1",
    prompt_function="prompt_fn_task1",
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task1",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=["loglikelihood_acc"],
)

task2 = LightevalTaskConfig(
    name="ger_rag_eval_task2",
    prompt_function="prompt_fn_task2",
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task2",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=["loglikelihood_acc"],
)

task3 = LightevalTaskConfig(
    name="ger_rag_eval_task3",
    prompt_function="prompt_fn_task3",
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task3",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=["loglikelihood_acc"],
)

task4 = LightevalTaskConfig(
    name="ger_rag_eval_task4",
    prompt_function="prompt_fn_task4",
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task4",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=["loglikelihood_acc"],
)

QUERY_TASK1: str = """\
Welche der folgenden Fragen (A oder B oder C oder D) lässt sich anhand des Kontext beantworten?

Kontext:
{context}

Fragen:
A: {choice_a}
B: {choice_b}
C: {choice_c}
D: {choice_d}
"""


QUERY_TASK2: str = """\
Auf Basis welcher der folgenden Kontexte (A oder B oder C oder D) lässt sich die Frage beantworten?

Frage: {question}

Kontexte:

A:
{choice_a}

B:
{choice_b}

C:
{choice_c}

D:
{choice_d}
"""


QUERY_TASK3: str = """\
Beantwortet die Antwort wirklich die Frage?
Antworte mit J für ja oder N für nein.

Die Frage: {question}

Die Antwort: {answer}
"""


QUERY_TASK4: str = """\
Lässt sich die Frage mithilfe der Informationen aus dem Kontext beantworten?
Antworte mit J für ja oder N für nein.

Kontext:
{context}

Die Frage: {question}
"""


def prompt_fn_task1(line, task_name: str = None):
    query = QUERY_TASK1.format(
        context=line["context"],
        choice_a=line["choice_a"],
        choice_b=line["choice_b"],
        choice_c=line["choice_c"],
        choice_d=line["choice_d"],
    )
    choices = ["A", "B", "C", "D"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def prompt_fn_task2(line, task_name: str = None):
    query = QUERY_TASK2.format(
        question=line["question"],
        choice_a=line["choice_a"],
        choice_b=line["choice_b"],
        choice_c=line["choice_c"],
        choice_d=line["choice_d"],
    )
    choices = ["A", "B", "C", "D"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def prompt_fn_task3(line, task_name: str = None):
    query = QUERY_TASK3.format(
        question=line["question"],
        answer=line["answer"],
    )
    choices = ["J", "N"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def prompt_fn_task4(line, task_name: str = None):
    query = QUERY_TASK4.format(
        question=line["question"],
        context=line["context"],
    )
    choices = ["J", "N"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


# STORE YOUR EVALS
_TASKS = [task1, task2, task3, task4]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
