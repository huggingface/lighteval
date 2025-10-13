# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


"""benchmark where we ask the model to unscramble a word, either anagram or
random insertion.
Don't remember where it's from.

https://huggingface.co/datasets/lighteval/GPT3_unscramble
"""

unscramble_anagrams1 = LightevalTaskConfig(
    name="unscramble:anagrams1",
    suite=["lighteval"],
    prompt_function=prompt.unscramble,
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
    suite=["lighteval"],
    prompt_function=prompt.unscramble,
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
    suite=["lighteval"],
    prompt_function=prompt.unscramble,
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
    suite=["lighteval"],
    prompt_function=prompt.unscramble,
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
    suite=["lighteval"],
    prompt_function=prompt.unscramble,
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
