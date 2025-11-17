"""
name:
GLUE

dataset:
nyu-mll/glue, aps/super_glue

abstract:
The General Language Understanding Evaluation (GLUE) benchmark is a collection
of resources for training, evaluating, and analyzing natural language
understanding systems.

languages:
english

tags:
classification, language-understanding

paper:
"""

import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def boolq_harness_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['passage']}\nQuestion: {line['question']}?\nAnswer:",
        choices=[" no", " yes"],
        gold_index=int(line["label"]),
    )


def cb_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['premise']}\nQuestion: {line['hypothesis']}. True, False or Neither?\nAnswer:",
        choices=[" True", " False", " Neither"],
        gold_index=int(line["label"]),
    )


def copa_prompt(line, task_name: str = None):
    connector = {"cause": "because", "effect": "therefore"}[line["question"]]
    return Doc(
        task_name=task_name,
        query=f"{line['premise'].strip()[:-1]} {connector}",
        choices=[f" {line[c][0].lower()}{line[c][1:]}" for c in ["choice1", "choice2"]],
        gold_index=int(line["label"]),
    )


def multirc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['paragraph']}\nQuestion: {line['question']}\nAnswer:",
        choices=[f" {line['answer']}\nIs the answer correct? yes", f" {line['answer']}\nIs the answer correct? no"],
        gold_index=0 if line["label"] else 1,
    )


def wic_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Sentence 1: {line['sentence1']}\nSentence 2: {line['sentence2']}\nQuestion: Is the word '{line['word']}' used in the same way in the two sentences above?\nAnswer:",
        choices=[" no", " yes"],
        gold_index=int(line["label"]),
    )


def wsc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['text']}\n'Question: In the passage above, does the pronoun {line['span2_text']} refer to {line['span1_text']}?\nAnswer:",
        choices=[" no", " yes"],
        gold_index=int(line["label"]),
    )


def cola_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['sentence']}\nQuestion: Does this sentence make sense?\nAnswer:",
        choices=[" no", " yes"],
        gold_index=int(line["label"]),
    )


def mnli_prompt(line, task_name: str = None):
    hypothesis = line["hypothesis"].strip() + ("" if line["hypothesis"].strip().endswith(".") else ".")
    return Doc(
        task_name=task_name,
        query=f"{line['premise']}\nQuestion: {hypothesis} True, False or Neither?\nAnswer:",
        choices=[" True", " Neither", " False"],
        gold_index=int(line["label"]),
    )


def mrpc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Sentence 1: {line['sentence1']}\nSentence 2: {line['sentence2']}\nQuestion: Do both sentences mean the same thing?\nAnswer:",
        choices=[" no", " yes"],
        gold_index=int(line["label"]),
    )


def qnli_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['question']}\n{line['sentence']}\nQuestion: Does this response answer the question?\nAnswer:",
        choices=[" yes", " no"],
        gold_index=int(line["label"]),
    )


def qqp_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question 1: {line['question1']}\nQuestion 2: {line['question2']}\nQuestion: Do both questions ask the same thing?\nAnswer:",
        choices=[" no", " yes"],
        gold_index=int(line["label"]),
    )


def rte_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['sentence1']}\nQuestion: {line['sentence2']} True or False?\nAnswer:",
        choices=[" True", " False"],
        gold_index=int(line["label"]),
    )


def sst_prompt(line, task_name: str = None):
    def general_detokenize(cur_string):
        cur_string = cur_string.replace(" n't", "n't")
        cur_string = cur_string.replace(" )", ")")
        cur_string = cur_string.replace("( ", "(")
        cur_string = cur_string.replace('" ', '"')
        cur_string = cur_string.replace(' "', '"')
        cur_string = re.sub(r" (['.,])", r"\1", cur_string)
        return cur_string

    return Doc(
        task_name=task_name,
        query=f"{general_detokenize(line['sentence'])}\nQuestion: Is this sentence positive or negative?\nAnswer:",
        choices=[" negative", " positive"],
        gold_index=int(line["label"]),
    )


def stsb_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"sentence 1: {line['sentence1']}\nsentence 2: {line['sentence2']}\nOn a scale of 0 to 5, how similar are the two sentences?\nAnswer:",
        gold_index=0,
        choices=[line["label"]],
    )


def wnli_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['sentence1']}\nQuestion: {line['sentence2']} True or False?\nAnswer:",
        choices=[" False", " True"],
        gold_index=int(line["label"]),
    )


glue_cola = LightevalTaskConfig(
    name="glue:cola",
    prompt_function=cola_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="cola",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.mcc],
    stop_sequence=["\n"],
    version=0,
)

glue_mnli = LightevalTaskConfig(
    name="glue:mnli",
    prompt_function=mnli_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="mnli_matched",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_mnli_mismatched = LightevalTaskConfig(
    name="glue:mnli_mismatched",
    prompt_function=mnli_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="mnli_mismatched",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_mrpc = LightevalTaskConfig(
    name="glue:mrpc",
    prompt_function=mrpc_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="mrpc",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
    stop_sequence=["\n"],
    version=0,
)

glue_qnli = LightevalTaskConfig(
    name="glue:qnli",
    prompt_function=qnli_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="qnli",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_qqp = LightevalTaskConfig(
    name="glue:qqp",
    prompt_function=qqp_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="qqp",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
    stop_sequence=["\n"],
    version=0,
)

glue_rte = LightevalTaskConfig(
    name="glue:rte",
    prompt_function=rte_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="rte",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_sst2 = LightevalTaskConfig(
    name="glue:sst2",
    prompt_function=sst_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="sst2",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_stsb = LightevalTaskConfig(
    name="glue:stsb",
    prompt_function=stsb_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="stsb",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

glue_wnli = LightevalTaskConfig(
    name="glue:wnli",
    prompt_function=wnli_prompt,
    hf_repo="nyu-mll/glue",
    hf_subset="wnli",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_boolq = LightevalTaskConfig(
    name="super_glue:boolq",
    prompt_function=boolq_harness_prompt,
    hf_repo="aps/super_glue",
    hf_subset="boolq",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_cb = LightevalTaskConfig(
    name="super_glue:cb",
    prompt_function=cb_prompt,
    hf_repo="aps/super_glue",
    hf_subset="cb",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.multi_f1_numeric],
    stop_sequence=["\n"],
    version=0,
)

super_glue_copa = LightevalTaskConfig(
    name="super_glue:copa",
    prompt_function=copa_prompt,
    hf_repo="aps/super_glue",
    hf_subset="copa",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_rte = LightevalTaskConfig(
    name="super_glue:rte",
    prompt_function=rte_prompt,
    hf_repo="aps/super_glue",
    hf_subset="rte",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_multirc = LightevalTaskConfig(
    name="super_glue:multirc",
    prompt_function=multirc_prompt,
    hf_repo="aps/super_glue",
    hf_subset="multirc",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_wic = LightevalTaskConfig(
    name="super_glue:wic",
    prompt_function=wic_prompt,
    hf_repo="aps/super_glue",
    hf_subset="wic",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

super_glue_wsc = LightevalTaskConfig(
    name="super_glue:wsc",
    prompt_function=wsc_prompt,
    hf_repo="aps/super_glue",
    hf_subset="wsc",
    hf_avail_splits=["test", "train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    glue_cola,
    glue_mnli,
    glue_mnli_mismatched,
    glue_mrpc,
    glue_qnli,
    glue_qqp,
    glue_rte,
    glue_sst2,
    glue_stsb,
    glue_wnli,
    super_glue_boolq,
    super_glue_cb,
    super_glue_copa,
    super_glue_rte,
    super_glue_multirc,
    super_glue_wic,
    super_glue_wsc,
]
