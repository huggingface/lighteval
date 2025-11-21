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

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def bbq_prompt(line, task_name: str = None):
    query = f"The following are multiple choice questions (with answers).\nPassage: {line['context']}\nQuestion: {line['question']}"
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(line["choices"])],
        gold_index=int(line["gold_index"]),
    )


def record_to_sample(record):
    query = f"{record['context']}\n{record['question']}"
    choices = record["choices"]
    target = ascii_uppercase[record["gold_index"]]
    return Sample(input=query, target=target, choices=choices)


bbq = LightevalTaskConfig(
    name="bbq",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Age = LightevalTaskConfig(
    name="bbq:Age",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Disability_status = LightevalTaskConfig(
    name="bbq:Disability_status",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Gender_identity = LightevalTaskConfig(
    name="bbq:Gender_identity",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Nationality = LightevalTaskConfig(
    name="bbq:Nationality",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Physical_appearance = LightevalTaskConfig(
    name="bbq:Physical_appearance",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Race_ethnicity = LightevalTaskConfig(
    name="bbq:Race_ethnicity",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Race_x_SES = LightevalTaskConfig(
    name="bbq:Race_x_SES",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Race_x_gender = LightevalTaskConfig(
    name="bbq:Race_x_gender",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Religion = LightevalTaskConfig(
    name="bbq:Religion",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_SES = LightevalTaskConfig(
    name="bbq:SES",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bbq_Sexual_orientation = LightevalTaskConfig(
    name="bbq:Sexual_orientation",
    prompt_function=bbq_prompt,
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
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
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
