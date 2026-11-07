"""
name:
HAE-RAE Bench

dataset:
HAERAE-HUB/HAE_RAE_BENCH_1.1

abstract:
HAE-RAE Bench is a Korean knowledge benchmark built from original Korean
sources rather than translated from English. This port covers the five
multiple-choice subsets of the original benchmark (standard nomenclature,
loan words, rare words, general knowledge, history; 1,091 questions), which are
the same subsets as the `haerae` group of EleutherAI lm-evaluation-harness.
`haerae` scores the answer label like lm-evaluation-harness does. `haerae_cf`
scores the text of each option appended to the question. The paper describes
its evaluation as the log-likelihood of each option concatenated to the
question, run with lm-evaluation-harness; it does not give the prompt text, so
the "질문: ... 정답:" wording here is ours.

languages:
korean

tags:
knowledge, multiple-choice

paper:
https://arxiv.org/abs/2309.02706
"""

import ast
import re

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Dataset commit pinned so scores stay comparable (main on 2026-11-03).
HAERAE_REVISION = "b480e81024913f27783d2b05d2f0b10089db19ad"

HAERAE_SUBSETS = ["standard_nomenclature", "loan_words", "rare_words", "general_knowledge", "history"]

_LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)"]


def haerae_prompt(line, task_name: str = None):
    # `query` already holds the instruction, the question, the lettered options and "### 정답:", as in
    # lm-eval's doc_to_text. lm-eval joins context and choice with target_delimiter=" ".
    return Doc(
        task_name=task_name,
        query=line["query"],
        choices=[f" {label}" for label in _LABELS],
        gold_index=_LABELS.index(line["answer"]),
    )


# The question block of `query` sits between "### 질문:" and "### 선택지:".
_QUESTION_RE = re.compile(r"### 질문:\s*(.*?)\s*### 선택지:", re.DOTALL)
_EMPTY_NOTE_RE = re.compile(r"^### 참고:nan[ \t]*\n?", re.MULTILINE)


def haerae_cf_prompt(line, task_name: str = None):
    # Closed-form: the question followed by the option text. The general_knowledge `query` can carry a
    # literal "### 참고:nan" line (an empty note, 98 of 176 questions); it is dropped here. The label prompt
    # keeps the `query` untouched. Notes with real text are kept.
    options = ast.literal_eval(line["options"])
    question = _EMPTY_NOTE_RE.sub("", _QUESTION_RE.search(line["query"]).group(1)).strip()
    return Doc(
        task_name=task_name,
        query=f"질문: {question}\n정답:",
        choices=[f" {option}" for option in options],
        gold_index=_LABELS.index(line["answer"]),
    )


def _task(name: str, prompt_function, subset: str, metrics: list) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=prompt_function,
        hf_repo="HAERAE-HUB/HAE_RAE_BENCH_1.1",
        hf_subset=subset,
        hf_revision=HAERAE_REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=metrics,
        version=0,
    )


HAERAE_TASKS = [
    _task(f"haerae:{subset}", haerae_prompt, subset, [Metrics.loglikelihood_acc]) for subset in HAERAE_SUBSETS
]

HAERAE_CF_TASKS = [
    _task(
        f"haerae_cf:{subset}",
        haerae_cf_prompt,
        subset,
        [Metrics.loglikelihood_acc, LogLikelihoodAccMetric(normalization=LogProbCharNorm())],
    )
    for subset in HAERAE_SUBSETS
]

TASKS_TABLE = HAERAE_TASKS + HAERAE_CF_TASKS
