"""
name:
Bbq

dataset:
lighteval/bbq_helm

abstract:
The Bias Benchmark for Question Answering (BBQ) for measuring social bias in
question answering in ambiguous and unambigous context .

languages:
english

tags:
bias, multiple-choice, qa

paper:
https://arxiv.org/abs/2110.08193
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


bbq = LightevalTaskConfig(
    name="bbq",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="all",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Age = LightevalTaskConfig(
    name="bbq:Age",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Age",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Disability_status = LightevalTaskConfig(
    name="bbq:Disability_status",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Disability_status",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Gender_identity = LightevalTaskConfig(
    name="bbq:Gender_identity",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Gender_identity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Nationality = LightevalTaskConfig(
    name="bbq:Nationality",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Nationality",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Physical_appearance = LightevalTaskConfig(
    name="bbq:Physical_appearance",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Physical_appearance",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Race_ethnicity = LightevalTaskConfig(
    name="bbq:Race_ethnicity",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Race_ethnicity",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Race_x_SES = LightevalTaskConfig(
    name="bbq:Race_x_SES",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Race_x_SES",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Race_x_gender = LightevalTaskConfig(
    name="bbq:Race_x_gender",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Race_x_gender",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Religion = LightevalTaskConfig(
    name="bbq:Religion",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Religion",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_SES = LightevalTaskConfig(
    name="bbq:SES",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="SES",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

bbq_Sexual_orientation = LightevalTaskConfig(
    name="bbq:Sexual_orientation",
    prompt_function=prompt.bbq,
    hf_repo="lighteval/bbq_helm",
    hf_subset="Sexual_orientation",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    bbq,
    bbq_Age,
    bbq_Disability_status,
    bbq_Gender_identity,
    bbq_Nationality,
    bbq_Physical_appearance,
    bbq_Race_ethnicity,
    bbq_Race_x_SES,
    bbq_Race_x_gender,
    bbq_Religion,
    bbq_SES,
    bbq_Sexual_orientation,
]
