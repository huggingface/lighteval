"""
name:
Ethics

dataset:
lighteval/hendrycks_ethics

abstract:
The Ethics benchmark for evaluating the ability of language models to reason about
ethical issues.

languages:
english

tags:
classification, ethics, justice, morality, utilitarianism, virtue

paper:
https://arxiv.org/abs/2008.02275
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


ethics_commonsense = LightevalTaskConfig(
    name="ethics:commonsense",
    prompt_function=prompt.ethics_commonsense,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="commonsense",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_deontology = LightevalTaskConfig(
    name="ethics:deontology",
    prompt_function=prompt.ethics_deontology,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="deontology",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_justice = LightevalTaskConfig(
    name="ethics:justice",
    prompt_function=prompt.ethics_justice,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="justice",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_utilitarianism = LightevalTaskConfig(
    name="ethics:utilitarianism",
    prompt_function=prompt.ethics_utilitarianism,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="utilitarianism",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ethics_virtue = LightevalTaskConfig(
    name="ethics:virtue",
    prompt_function=prompt.ethics_virtue,
    hf_repo="lighteval/hendrycks_ethics",
    hf_subset="virtue",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    ethics_commonsense,
    ethics_deontology,
    ethics_justice,
    ethics_utilitarianism,
    ethics_virtue,
]
