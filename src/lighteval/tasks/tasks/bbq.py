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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
    suite=["lighteval"],
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
