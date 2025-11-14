"""
name:
Lexglue

dataset:
lighteval/lexglue

abstract:
LexGLUE: A Benchmark Dataset for Legal Language Understanding in English

languages:
english

tags:
classification, legal

paper:
https://arxiv.org/abs/2110.00976
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def lex_glue(line, instruction, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{instruction}\nPassage: {line['input']}\nAnswer: ",
        choices=line["references"],
        gold_index=[line["references"].index(item) for item in line["gold"]],
        instruction=instruction + "\n",
    )


def lex_glue_ecthr_a_prompt(line, task_name: str = None):
    instruction = "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). Predict the articles of the ECtHR that were violated (if any)."
    return lex_glue(line, instruction, task_name)


def lex_glue_ecthr_b_prompt(line, task_name: str = None):
    instruction = "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). Predict the articles of ECtHR that were allegedly violated (considered by the court)."
    return lex_glue(line, instruction, task_name)


def lex_glue_scotus_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a case heard at the Supreme Court of the United States (SCOTUS). Predict the relevant issue area."
    return lex_glue(line, instruction, task_name)


def lex_glue_eurlex_prompt(line, task_name: str = None):
    instruction = "In this task, you are given an EU law document published in the EUR-Lex portal. Predict the relevant EuroVoc concepts."
    return lex_glue(line, instruction, task_name)


def lex_glue_ledgar_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a contract provision \nfrom contracts obtained from US Securities and Exchange Commission (SEC) filings. Predict the main topic."
    return lex_glue(line, instruction, task_name)


def lex_glue_unfair_tos_prompt(line, task_name: str = None):
    instruction = "In this task, you are given a sentence \nfrom a Terms of Service (ToS) document from on-line platforms. Predict the types of unfair contractual terms"
    return lex_glue(line, instruction, task_name)


def lex_glue_case_hold_prompt(line, task_name: str = None):
    instruction = "In this task, you are given an excerpt from a court decision, \ncontaining a reference to a particular case, while the holding statement is masked out. Predict the index of the holding statement fitting in the context at <HOLDING> from a selection of five choices."
    return lex_glue(line, instruction, task_name)


lexglue_case_hold = LightevalTaskConfig(
    name="lexglue:case_hold",
    prompt_function=lex_glue_case_hold_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="case_hold",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_ecthr_a = LightevalTaskConfig(
    name="lexglue:ecthr_a",
    prompt_function=lex_glue_ecthr_a_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="ecthr_a",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_ecthr_b = LightevalTaskConfig(
    name="lexglue:ecthr_b",
    prompt_function=lex_glue_ecthr_b_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="ecthr_b",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_eurlex = LightevalTaskConfig(
    name="lexglue:eurlex",
    prompt_function=lex_glue_eurlex_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="eurlex",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_ledgar = LightevalTaskConfig(
    name="lexglue:ledgar",
    prompt_function=lex_glue_ledgar_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="ledgar",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_scotus = LightevalTaskConfig(
    name="lexglue:scotus",
    prompt_function=lex_glue_scotus_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="scotus",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_unfair_tos = LightevalTaskConfig(
    name="lexglue:unfair_tos",
    prompt_function=lex_glue_unfair_tos_prompt,
    hf_repo="lighteval/lexglue",
    hf_subset="unfair_tos",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    lexglue_case_hold,
    lexglue_ecthr_a,
    lexglue_ecthr_b,
    lexglue_eurlex,
    lexglue_ledgar,
    lexglue_scotus,
    lexglue_unfair_tos,
]
