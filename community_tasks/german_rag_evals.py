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

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
This module implements the 4 tasks of deutsche-telekom/Ger-RAG-eval.
See: https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_fn_choose_question_by_context(line, task_name: str = None):
    instruction = "Welche der folgenden Fragen (A oder B oder C oder D) lässt sich anhand des Kontext beantworten?\n\n"
    query_template = """\
Kontext:
{context}

Fragen:
A: {choice_a}
B: {choice_b}
C: {choice_c}
D: {choice_d}

Antwort:"""
    query = instruction + query_template.format(
        context=line["context"],
        choice_a=line["choice_a"],
        choice_b=line["choice_b"],
        choice_c=line["choice_c"],
        choice_d=line["choice_d"],
    )
    choices = ["A", "B", "C", "D"]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def prompt_fn_choose_context_by_question(line, task_name: str = None):
    instruction = (
        "Auf Basis welcher der folgenden Kontexte (A oder B oder C oder D) lässt sich die Frage beantworten?\n\n"
    )
    query_template = """\
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

Antwort:"""
    query = instruction + query_template.format(
        question=line["question"],
        choice_a=line["choice_a"],
        choice_b=line["choice_b"],
        choice_c=line["choice_c"],
        choice_d=line["choice_d"],
    )
    choices = ["A", "B", "C", "D"]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def prompt_fn_question_answer_match(line, task_name: str = None):
    instruction = "Beantwortet die Antwort wirklich die Frage? Antworte mit J für ja oder N für nein.\n\n"
    query_template = """\
Die Frage: {question}

Die Antwort: {answer}

Auswahl (J/N):"""
    query = instruction + query_template.format(
        question=line["question"],
        answer=line["answer"],
    )
    choices = ["J", "N"]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def prompt_fn_context_question_match(line, task_name: str = None):
    instruction = "Lässt sich die Frage mithilfe der Informationen aus dem Kontext beantworten? Antworte mit J für ja oder N für nein.\n\n"
    query_template = """\
Kontext:
{context}

Die Frage: {question}

Auswahl (J/N):"""
    query = instruction + query_template.format(
        question=line["question"],
        context=line["context"],
    )
    choices = ["J", "N"]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


# Task 1: Choose question by context.
# Given is a context and 4 questions.
# The task is to decide which question can be answered by the context.
task1 = LightevalTaskConfig(
    name="german_rag_eval:choose_question_by_context",
    prompt_function=prompt_fn_choose_question_by_context,
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task1",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc],
    version=1,
)

# Task 2: Choose context by question.
# Given is a question and 4 contexts.
# The task is to decide which context can answer the question.
task2 = LightevalTaskConfig(
    name="german_rag_eval:choose_context_by_question",
    prompt_function=prompt_fn_choose_context_by_question,
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task2",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc],
    version=1,
)


# Task 3: Question-answer match.
# Given is a question and an answer.
# The task is to decide whether the answer actualy answers the question.
task3 = LightevalTaskConfig(
    name="german_rag_eval:question_answer_match",
    prompt_function=prompt_fn_question_answer_match,
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task3",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc],
    version=1,
)

# Task 4: Context-question match.
# Given is a context and a question.
# The task is to decide whether the question can be answered by the context or not.
task4 = LightevalTaskConfig(
    name="german_rag_eval:context_question_match",
    prompt_function=prompt_fn_context_question_match,
    suite=["community"],
    hf_repo="deutsche-telekom/Ger-RAG-eval",
    hf_subset="task4",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc],
    version=1,
)


# STORE YOUR EVALS
TASKS_TABLE = [task1, task2, task3, task4]
