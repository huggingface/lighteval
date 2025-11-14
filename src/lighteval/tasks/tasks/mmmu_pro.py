"""
name:
Mmmu Pro

dataset:
MMMU/MMMU_pro

abstract:

languages:
english

tags:
general-knowledge, knowledge, multimodal, multiple-choice

paper:
https://arxiv.org/abs/2409.02813
"""

import ast
import re
import string

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mmmu_pro_prompt(line, task_name: str = None):
    question = line["question"]
    choices_string = line["options"]
    answer = line["answer"]

    instructions = "Answer with the option letter from the given choices directly."

    choices = ast.literal_eval(str(choices_string))
    choices_letters = [chr(ord("A") + i) for i in range(len(choices))]
    choices = [f"{letter}. {choice}" for letter, choice in zip(choices_letters, choices)]

    formatted_choices = "\n".join(choices)
    prompt_text = f"\n{question}\n{formatted_choices}"

    image_order = []
    for num in re.findall(r"<image\s+(\d+)>", prompt_text):
        num = int(num)
        if num not in image_order:
            image_order.append(num)
    images = [line[f"image_{i}"].convert("RGB") for i in image_order]

    gold_index = string.ascii_uppercase.index(answer)

    prompt_text = re.sub(r"<image\s+(\d+)>", "[image \\1]", prompt_text)
    choices = [re.sub(r"<image\s+(\d+)>", "[image \\1]", choice) for choice in choices]

    return Doc(
        task_name=task_name,
        query=prompt_text,
        choices=choices,
        gold_index=gold_index,
        images=images,
        specific={"id": line["id"]},
        instruction=instructions,
    )


def mmmu_pro_vision_prompt(line, task_name: str = None):
    instruction = (
        "Answer with the option letter from the given choices directly."
        " The last line of your response should be of the following format: "
        "'Answer: $LETTER' (without quotes) where LETTER is one of options."
    )

    choices_string = line["options"]
    choices = ast.literal_eval(str(choices_string))
    choices_letters = [chr(ord("A") + i) for i in range(len(choices))]
    choices = [f"{letter}. {choice}" for letter, choice in zip(choices_letters, choices)]

    answer = line["answer"]
    gold_index = string.ascii_uppercase.index(answer)

    images = [line["image"]]

    return Doc(
        task_name=task_name,
        query=instruction,
        choices=choices,
        gold_index=gold_index,
        images=images,
        instruction=instruction,
    )


mmmu_pro_standard_4_options = LightevalTaskConfig(
    name="mmmu_pro:standard-4",
    prompt_function=mmmu_pro_prompt,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="standard (4 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


mmmu_pro_standard_10_options = LightevalTaskConfig(
    name="mmmu_pro:standard-10",
    prompt_function=mmmu_pro_prompt,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="standard (10 options)",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


mmmu_pro_vision = LightevalTaskConfig(
    name="mmmu_pro:vision",
    prompt_function=mmmu_pro_vision_prompt,
    hf_repo="MMMU/MMMU_pro",
    hf_subset="vision",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=30,  # expected an answer in a format 'Answer: B'
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=0,
)


TASKS_TABLE = [
    mmmu_pro_standard_4_options,
    mmmu_pro_standard_10_options,
    mmmu_pro_vision,
]
