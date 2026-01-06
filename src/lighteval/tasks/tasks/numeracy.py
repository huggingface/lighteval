"""
name:
Numeracy

dataset:
lighteval/numeracy

abstract:
Numeracy is a benchmark for evaluating the ability of language models to reason about mathematics.

languages:
english

tags:
math, reasoning

paper:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


numeracy_vars_names = ["x", "y", "z"]


def numeracy_prompt(line, task_name: str = None):
    vars = ""
    for ix, value in enumerate(line["vars"]):
        vars += f"{numeracy_vars_names[ix]} {value}, "
    vars += numeracy_vars_names[ix + 1]
    return Doc(task_name=task_name, query=f"{line['equation']}, {vars}", gold_index=0, choices=[str(line["output"])])


numeracy_linear_example = LightevalTaskConfig(
    name="numeracy:linear_example",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="linear_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_linear_standard = LightevalTaskConfig(
    name="numeracy:linear_standard",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="linear_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_parabola_example = LightevalTaskConfig(
    name="numeracy:parabola_example",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="parabola_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_parabola_standard = LightevalTaskConfig(
    name="numeracy:parabola_standard",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="parabola_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_paraboloid_example = LightevalTaskConfig(
    name="numeracy:paraboloid_example",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="paraboloid_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_paraboloid_standard = LightevalTaskConfig(
    name="numeracy:paraboloid_standard",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="paraboloid_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_plane_example = LightevalTaskConfig(
    name="numeracy:plane_example",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="plane_example",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

numeracy_plane_standard = LightevalTaskConfig(
    name="numeracy:plane_standard",
    prompt_function=numeracy_prompt,
    hf_repo="lighteval/numeracy",
    hf_subset="plane_standard",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    numeracy_linear_example,
    numeracy_linear_standard,
    numeracy_parabola_example,
    numeracy_parabola_standard,
    numeracy_paraboloid_example,
    numeracy_paraboloid_standard,
    numeracy_plane_example,
    numeracy_plane_standard,
]
