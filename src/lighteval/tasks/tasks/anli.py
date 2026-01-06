"""
name:
Anli

dataset:
facebook/anli

abstract:
The Adversarial Natural Language Inference (ANLI) is a new large-scale NLI
benchmark dataset, The dataset is collected via an iterative, adversarial
human-and-model-in-the-loop procedure. ANLI is much more difficult than its
predecessors including SNLI and MNLI. It contains three rounds. Each round has
train/dev/test splits.

languages:
english

tags:
nli, reasoning

paper:
https://arxiv.org/abs/1910.14599
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def anli_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['premise']}\nQuestion: {line['hypothesis']} True, False, or Neither?\nAnswer:",
        choices=[" True", " Neither", " False"],
        gold_index=int(line["label"]),
    )


def record_to_sample(record):
    choices = ["True", "Neither", "False"]
    query = f"{record['premise']}\nQuestion: {record['hypothesis']}"
    return Sample(input=query, target=ascii_uppercase[record["label"]], choices=choices)


anli_r1 = LightevalTaskConfig(
    name="anli:r1",
    prompt_function=anli_prompt,
    hf_repo="facebook/anli",
    hf_subset="plain_text",
    hf_avail_splits=["train_r1", "dev_r1", "test_r1"],
    evaluation_splits=["test_r1"],
    few_shots_split="train_r1",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


anli_r2 = LightevalTaskConfig(
    name="anli:r2",
    prompt_function=anli_prompt,
    hf_repo="facebook/anli",
    hf_subset="plain_text",
    hf_avail_splits=["train_r2", "dev_r2", "test_r2"],
    evaluation_splits=["test_r2"],
    few_shots_split="train_r2",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


anli_r3 = LightevalTaskConfig(
    name="anli:r3",
    prompt_function=anli_prompt,
    hf_repo="facebook/anli",
    hf_subset="plain_text",
    hf_avail_splits=["train_r3", "dev_r3", "test_r3"],
    evaluation_splits=["test_r3"],
    few_shots_split="train_r3",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    anli_r1,
    anli_r2,
    anli_r3,
]
