"""
name:
Unscramble

dataset:
lighteval/GPT3_unscramble

abstract:
Benchmark where we ask the model to unscramble a word, either anagram or
random insertion.

languages:
english

tags:
language-modeling, reasoning

paper:
https://huggingface.co/datasets/lighteval/GPT3_unscramble
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def unscramble_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, query=line["context"], gold_index=0, choices=[line["completion"]])


unscramble_anagrams1 = LightevalTaskConfig(
    name="unscramble:anagrams1",
    prompt_function=unscramble_prompt,
    hf_repo="lighteval/GPT3_unscramble",
    hf_subset="default",
    hf_avail_splits=["mid_word_1_anagrams"],
    evaluation_splits=["mid_word_1_anagrams"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

unscramble_anagrams2 = LightevalTaskConfig(
    name="unscramble:anagrams2",
    prompt_function=unscramble_prompt,
    hf_repo="lighteval/GPT3_unscramble",
    hf_subset="default",
    hf_avail_splits=["mid_word_2_anagrams"],
    evaluation_splits=["mid_word_2_anagrams"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

unscramble_cycle_letters = LightevalTaskConfig(
    name="unscramble:cycle_letters",
    prompt_function=unscramble_prompt,
    hf_repo="lighteval/GPT3_unscramble",
    hf_subset="default",
    hf_avail_splits=["cycle_letters_in_word"],
    evaluation_splits=["cycle_letters_in_word"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

unscramble_random_insertion = LightevalTaskConfig(
    name="unscramble:random_insertion",
    prompt_function=unscramble_prompt,
    hf_repo="lighteval/GPT3_unscramble",
    hf_subset="default",
    hf_avail_splits=["random_insertion_in_word"],
    evaluation_splits=["random_insertion_in_word"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

unscramble_reversed_words = LightevalTaskConfig(
    name="unscramble:reversed_words",
    prompt_function=unscramble_prompt,
    hf_repo="lighteval/GPT3_unscramble",
    hf_subset="default",
    hf_avail_splits=["reversed_words"],
    evaluation_splits=["reversed_words"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    unscramble_anagrams1,
    unscramble_anagrams2,
    unscramble_cycle_letters,
    unscramble_random_insertion,
    unscramble_reversed_words,
]
