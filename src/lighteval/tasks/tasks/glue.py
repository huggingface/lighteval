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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


glue_cola = LightevalTaskConfig(
    name="glue:cola",
    prompt_function=prompt.cola,
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
    prompt_function=prompt.mnli,
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
    prompt_function=prompt.mnli,
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
    prompt_function=prompt.mrpc,
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
    prompt_function=prompt.qnli,
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
    prompt_function=prompt.qqp,
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
    prompt_function=prompt.rte,
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
    prompt_function=prompt.sst,
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
    prompt_function=prompt.stsb,
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
    prompt_function=prompt.wnli,
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
    prompt_function=prompt.boolq_harness,
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
    prompt_function=prompt.cb,
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
    prompt_function=prompt.copa,
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
    prompt_function=prompt.rte,
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
    prompt_function=prompt.multirc,
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
    prompt_function=prompt.wic,
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
    prompt_function=prompt.wsc,
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
